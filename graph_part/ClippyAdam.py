import torch
from torch.optim.optimizer import Optimizer


def shrink_by_references(tensor, references, relative_factors, absolute_factor):
    if any(relative_factor < 0 for relative_factor in relative_factors):
        raise ValueError("relative_factors must all be non-negative.")
    if absolute_factor < 0:
        raise ValueError("absolute_factor must be non-negative.")
    if len(references) != len(relative_factors):
        raise ValueError(
            "references and relative_factors must have the same length. "
            f"Instead they are {len(references)} and {len(relative_factors)}.")

    max_delta = sum(
        (torch.abs(reference) * relative_factor
         for reference, relative_factor in zip(references, relative_factors)),
        torch.tensor(absolute_factor))

    per_element_scale = torch.where(
        tensor == 0., torch.tensor(1.), max_delta / torch.abs(tensor))

    scale = torch.min(torch.tensor(1.), torch.min(per_element_scale))

    return tensor * scale, scale


class ClippyAdam(Optimizer):
    def __init__(
            self,
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            initial_accumulator_value=0.01,
            variable_relative_threshold=0.5,
            accumulator_relative_threshold=0.0,
            absolute_threshold=1e-2,
            export_clipping_factors=False,
            clip_accumulator_update=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
            variable_relative_threshold=variable_relative_threshold,
            accumulator_relative_threshold=accumulator_relative_threshold,
            absolute_threshold=absolute_threshold,
            export_clipping_factors=export_clipping_factors,
            clip_accumulator_update=clip_accumulator_update,
        )

        super(ClippyAdam, self).__init__(params, defaults)

        self._var_dict = {id(p): p for p in self.param_groups[0]['params']}

        self._m = []
        self._v = []
        self.clipping_factors = []

        for p in self.param_groups[0]['params']:
            self._m.append(torch.zeros_like(p))
            self._v.append(torch.full_like(p, initial_accumulator_value))
            if export_clipping_factors:
                self.clipping_factors.append(torch.empty(()))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]   # 之前的step累计数据

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = self._m[i]
                    state['v'] = self._v[i]
                    if group['export_clipping_factors']:
                        state['clipping_factor'] = self.clipping_factors[i]

                lr = group['lr']
                betas = group['betas']
                eps = group['eps']
                m = state['m']
                v = state['v']

                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                m_hat = m / (1 - betas[0] ** (state['step'] + 1))
                v_hat = v / (1 - betas[1] ** (state['step'] + 1))

                delta = lr * m_hat / (torch.sqrt(v_hat) + eps)

                references = [p.data, torch.sqrt(v_hat)]
                relative_factors = [group['variable_relative_threshold'], group['accumulator_relative_threshold']]
                absolute_factor = group['absolute_threshold']

                clipped_delta, clipping_factor = shrink_by_references(
                    delta,
                    references=references,
                    relative_factors=relative_factors,
                    absolute_factor=absolute_factor
                )

                if group['export_clipping_factors']:
                    state['clipping_factor'][:] = clipping_factor

                p.data.sub_(clipped_delta)

                state['step'] += 1

        return loss




