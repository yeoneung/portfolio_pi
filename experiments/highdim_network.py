"""Continuous certificates on an irregular factor tree with correlated assets.

The deployed rule is explicitly componentwise, not the covariance-metric
greedy optimizer. Diagonal quadratic-form bounds enclose the coupled Bellman
Hamiltonian and the exact fixed-policy generator in the original coordinates.
Every coordinate has its own polynomial and a complete continuous cover.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from mpmath import iv

import highdim_local_verify as arithmetic

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model(d, correlation='1/10'):
    if d < 4:
        raise ValueError('At least four states are required')
    adjacency = [[F(0) for _ in range(d)] for _ in range(d)]
    for i in range(1, d):
        j = (i-1)//2
        adjacency[i][j] = adjacency[j][i] = F(1)
    degrees = [sum(row) for row in adjacency]
    weights = [[a/4 for a in row] for row in adjacency]
    for i in range(d):
        weights[i][i] = 1-degrees[i]/4
    covariance = [[F(i == j)+F(correlation)*adjacency[i][j]
                   for j in range(d)] for i in range(d)]
    common = dict(kappa='1', drift_neighbor='1/5', variance='1/16',
                  cross='1/8', cap='11/100', risk_free_rate='3/100',
                  horizon='1', radius='3/2', gamma='3')
    blocks = []
    for i in range(d):
        s = F(2*i, d-1)-1
        t = F(2*((3*i+1) % d), d-1)-1
        m = dict(common, lambda_constant=str(F(1, 5)*(1+s/5)),
                 lambda_self=str(F(1, 2)*(1+t/5)),
                 lambda_neighbor=str(F(1, 4)*(1-s/5)),
                 risk_lower=str(1-F(correlation)*degrees[i]),
                 risk_upper=str(1+F(correlation)*degrees[i]))
        blocks.append(dict(model=m, dimension=d, assets=d))
    return dict(dimension=d, assets=d, graph='undirected binary tree, parent(i)=(i-1)//2',
                covariance_correlation_exact=str(F(correlation)),
                network_weights=[[str(v) for v in row] for row in weights],
                asset_covariance=[[str(v) for v in row] for row in covariance],
                blocks=blocks,
                policy='componentwise clipped signal / gamma; zero outside cube; re-entry unchanged',
                evaluation='coordinate-dependent additive log-quadratics',
                certified=False)


def validate(candidate):
    d, blocks = candidate['dimension'], candidate['blocks']
    if len(blocks) != d or candidate['assets'] != d:
        raise ValueError('One block per state and asset is required')
    P = [[F(x) for x in row] for row in candidate['network_weights']]
    S = [[F(x) for x in row] for row in candidate['asset_covariance']]
    if len(P) != d or len(S) != d or any(len(x) != d for x in P+S):
        raise ValueError('Matrix dimensions disagree')
    if any(min(row) < 0 or sum(row) != 1 for row in P):
        raise ValueError('The network matrix must be stochastic')
    if any(P[i][j] != P[j][i] or S[i][j] != S[j][i] for i in range(d) for j in range(d)):
        raise ValueError('Symmetric matrices are required for this localization')
    variable = {'lambda_constant', 'lambda_self', 'lambda_neighbor', 'risk_lower', 'risk_upper'}
    for i, block in enumerate(blocks):
        m = {k: F(v) for k, v in block['model'].items()}
        if block['dimension'] != d or block['assets'] != d:
            raise ValueError('Block dimension mismatch')
        if any(v != blocks[0]['model'][k] for k, v in block['model'].items() if k not in variable):
            raise ValueError('Common state dynamics, risk aversion, and caps are required')
        off = sum(abs(S[i][j]) for j in range(d) if j != i)
        if S[i][i] != 1 or m['risk_lower'] != 1-off or m['risk_upper'] != 1+off:
            raise ValueError('Incorrect diagonal quadratic-form bounds')
        if m['risk_lower'] <= m['cross']**2/m['variance']:
            raise ValueError('Joint asset/state covariance is not certified positive definite')
        if not m['gamma'] > 1 or m['horizon'] != 1 or not m['kappa'] > m['drift_neighbor'] >= 0:
            raise ValueError('Unsupported risk aversion, horizon, or drift')
        if min(m['cap'], m['radius'], m['variance']) <= 0 or min(m['risk_free_rate'], m['cross']) < 0:
            raise ValueError('Unsupported diffusion or parameter signs')
        if m['lambda_self'] <= abs(m['lambda_neighbor']):
            raise ValueError('Predictor loading must be strictly diagonally dominant')
        for side in ('lower', 'upper'):
            if side in block:
                coeff = block[side]['potential']
                if len(block['base']) != 3*block['time_terms'] or len(block[side]['coeff']) != len(block['base']):
                    raise ValueError('Polynomial dimensions disagree')
                if coeff != blocks[0][side]['potential']:
                    raise ValueError('The network Jensen potential must be common')
                k = block['time_terms']
                if len(coeff) != 2*k:
                    raise ValueError('Potential dimension mismatch')
                quadratic = list(map(F, coeff[k:]))
                if any(quadratic[1:]) or (quadratic[0] > 0 if side == 'lower' else quadratic[0] < 0):
                    raise ValueError('Incorrect Jensen curvature for the network potential')
    return blocks


def train(candidate, degree=5, epochs=1200, seed=1110, device='cuda'):
    import torch
    from scipy.linalg import solve_continuous_lyapunov
    from highdim_local_fit import basis, rounded
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = torch.Generator(device=device).manual_seed(seed)
    d = candidate['dimension']
    ms = [{k: float(F(v)) for k, v in b['model'].items()} for b in candidate['blocks']]
    m = ms[0]
    matrix = lambda key: np.array([[float(F(x)) for x in row] for row in candidate[key]])
    P, S = matrix('network_weights'), matrix('asset_covariance')
    K = m['kappa']*np.eye(d)-m['drift_neighbor']*P
    stationary = solve_continuous_lyapunov(K, m['variance']*np.eye(d))
    tensor = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    chol, pt, st = tensor(np.linalg.cholesky(stationary)), tensor(P), tensor(S)
    ell, aa, bb = [tensor([v[key] for v in ms]) for key in
                   ('lambda_constant', 'lambda_self', 'lambda_neighbor')]
    coeff = torch.zeros((d, 3, degree), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([coeff], lr=.001)
    for epoch in range(epochs):
        tau = torch.rand(2048, device=device, dtype=torch.float64, generator=rng)
        z = torch.randn((2048, d), device=device, dtype=torch.float64, generator=rng)@chol.T
        broad = 2*m['radius']*torch.rand((2048, d), device=device, dtype=torch.float64, generator=rng)-m['radius']
        mask = torch.rand(2048, device=device, dtype=torch.float64, generator=rng) < .03
        z = torch.where(mask[:, None], broad, z)
        y = z@pt.T
        _, bt, dt = basis(tau, degree, torch)
        f, ft = torch.einsum('nk,ijk->nij', bt, coeff), torch.einsum('nk,ijk->nij', dt, coeff)
        grad = f[:, :, 1]+2*f[:, :, 2]*z
        signal = ell+aa*z+bb*y+m['cross']/np.sqrt(d)*grad
        control = torch.clamp(signal/m['gamma'], -m['cap'], m['cap'])
        core = ft[:, :, 0]+ft[:, :, 1]*z+ft[:, :, 2]*z*z
        core += (m['kappa']*z-m['drift_neighbor']*y)*grad
        core -= m['variance']/2*(2*f[:, :, 2]+grad*grad/d)
        residual = core.mean(1)+(m['gamma']-1)*(control*signal-m['gamma']/2*control*(control@st)).mean(1)
        loss = residual.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    for block, c in zip(candidate['blocks'], coeff.detach().cpu().numpy()):
        block.update(base=rounded(c.ravel()), time_terms=degree)
    candidate['training'] = dict(seconds=time.perf_counter()-start, epochs=epochs,
                                 seed=seed, device=device, final_sampled_loss=float(loss.detach()),
                                 gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None)


def fit_block(block, refinement=1, potentials=None, fit_potentials=False):
    from scipy.optimize import linprog
    from highdim_local_fit import basis, design, rounded
    start = time.perf_counter()
    d, k = block['dimension'], block['time_terms']
    m = {key: float(F(v)) for key, v in block['model'].items()}
    base = np.array([float(F(v)) for v in block['base']])
    ts = (1-np.cos(np.linspace(0, np.pi, 12*refinement+1)))/2
    zs = np.unique(np.r_[np.linspace(-m['radius'], m['radius'], 24*refinement+1),
                         np.linspace(-.4, .4, 12*refinement+1)])
    tt, xx, yy = np.meshgrid(ts, zs, zs, indexing='ij')
    t, x, y = tt.ravel(), xx.ravel(), yy.ravel()
    _, vt, vx, vxx, _ = design(t, x, k)
    tu, xu = np.meshgrid(ts, zs, indexing='ij')
    _, _, dx, _, trace = design(tu.ravel(), xu.ravel(), k)
    lam = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y
    g, h = m['cross']/np.sqrt(d), m['gamma']-1
    fixed = np.clip((lam+g*(vx@base))/m['gamma'], -m['cap'], m['cap'])
    cost = np.r_[np.ones(k), np.zeros(2*k)]
    pot = np.concatenate((basis(t, k)[0]*(y-x)[:, None], (y*y-x*x)[:, None]), axis=1)
    logs = []
    for side in ('lower', 'upper'):
        prev, trust = base.copy(), .1
        risk = m['risk_lower' if side == 'lower' else 'risk_upper']
        potential = np.zeros(k+1) if potentials is None else np.array(
            [float(F(v)) for v in potentials[side][:k+1]])
        for iteration in range(3):
            grad = vx@prev
            signal = lam+g*grad
            policy = np.clip(signal/(m['gamma']*risk), -m['cap'], m['cap']) if side == 'lower' else fixed
            residual = vt@prev+(m['kappa']*x-m['drift_neighbor']*y)*grad
            residual -= m['variance']/2*(vxx@prev+grad*grad/d)
            residual += h*(policy*signal-m['gamma']*risk/2*policy*policy)
            op = vt+(m['kappa']*x-m['drift_neighbor']*y-m['variance']/d*grad+h*g*policy)[:, None]*vx-m['variance']/2*vxx
            rhs = op@prev-residual
            if fit_potentials:
                op = np.concatenate((op, pot), axis=1)
                trace_matrix = np.pad(trace, ((0, 0), (0, k+1)))
                derivative_matrix = np.pad(dx, ((0, 0), (0, k+1)))
                costs = np.r_[cost, np.zeros(k+1)]
                bounds = [(-3., 3.)]*(len(base)+k)+([(-3., 0.)] if side == 'lower' else [(0., 3.)])
            else:
                rhs -= pot@potential
                trace_matrix, derivative_matrix, costs = trace, dx, cost
                bounds = [(-3., 3.)]*len(base)
            if side == 'lower':
                aa, bb = np.concatenate((op, trace_matrix)), np.r_[rhs-1e-7, trace@base]
                objective = -costs
            else:
                aa = np.concatenate((-op, -trace_matrix, derivative_matrix, -derivative_matrix))
                bb = np.r_[-rhs-1e-7-m['variance']/(2*d)*trust**2, -trace@base,
                           dx@prev+trust, -dx@prev+trust]
                objective = costs
            sol = linprog(objective, A_ub=aa, b_ub=bb, bounds=bounds, method='highs')
            if not sol.success:
                raise ArithmeticError(f'{side}: {sol.message}')
            prev = sol.x[:len(base)]
            if fit_potentials:
                potential = sol.x[len(base):]
            logs.append(dict(side=side, iteration=iteration, objective=float(sol.fun)))
        block[side] = dict(coeff=rounded(prev), potential=rounded(np.r_[potential, np.zeros(k-1)]))
    block['fit'] = dict(seconds=time.perf_counter()-start, refinement=refinement, diagnostics=logs)


def residual_polys(block):
    P, scalar = arithmetic.Poly, arithmetic.scalar
    d = block['dimension']
    m = {k: F(v) for k, v in block['model'].items()}
    base, lower, upper = [arithmetic.factor(c) for c in
                          (block['base'], block['lower']['coeff'], block['upper']['coeff'])]
    x, y = P.variable(1), P.variable(2)
    lam = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y
    cross = scalar(m['cross'])/iv.sqrt(scalar(d))
    base_signal = lam+base.derivative(1)*cross
    arrays, signals, thresholds = [], [], []
    for side, f in [('lower', lower), ('upper', upper)]:
        grad = f.derivative(1)
        core = f.derivative(0)+(m['kappa']*x-m['drift_neighbor']*y)*grad
        core -= m['variance']/2*grad.derivative(1)+m['variance']/F(2*d)*grad*grad
        core += arithmetic.potential(block[side]['potential'])
        signal = lam+grad*cross
        risk = m['risk_lower' if side == 'lower' else 'risk_upper']
        sig = signal if side == 'lower' else base_signal
        divisor = m['gamma']*(risk if side == 'lower' else 1)
        policies = [P.constant(-m['cap']), sig*(1/divisor), P.constant(m['cap'])]
        arrays.extend(core+(m['gamma']-1)*(p*signal-m['gamma']*risk/2*p*p) for p in policies)
        signals.append(sig)
        thresholds.append(divisor*m['cap'])
    arrays += signals+[(lower-base).divide_time(), (base-upper).divide_time(), base]
    return arrays, thresholds, m


def cell_bounds(lo, hi, thresholds):
    low, high = lo.min(axis=(1, 2, 3)), hi.max(axis=(1, 2, 3))
    extrema = []
    for i, threshold in enumerate(thresholds):
        tl, th = arithmetic.ends(arithmetic.scalar(threshold))
        a, b = low[6+i], high[6+i]
        possible = (a <= -tl, a <= th and b >= -th, b >= tl)
        indices = [3*i+j for j in range(3) if possible[j]]
        if not indices:
            raise ArithmeticError('No compatible clipping branch')
        extrema.append((min(low[j] for j in indices), max(high[j] for j in indices)))
    return dict(lower_residual_upper=float(extrema[0][1]), upper_residual_lower=float(extrema[1][0]),
                lower_trace_rate_upper=float(high[8]), upper_trace_rate_upper=float(high[9]),
                base_factor_exponent_upper=float(high[10]))


def verify_block(block, tolerance=F(1, 1000), max_depth=33, max_cells=1000000):
    start = time.perf_counter()
    polys, thresholds, m = residual_polys(block)
    degrees = tuple(max(key[i] for p in polys for key in p.c) for i in range(3))
    raw = [arithmetic.bernstein(p, m['radius'], degrees) for p in polys]
    stack = [(np.stack([x[0] for x in raw]), np.stack([x[1] for x in raw]), 0)]
    worst, volume, accepted, visited, deepest = None, F(0), 0, 0, 0
    while stack:
        lo, hi, depth = stack.pop()
        visited += 1
        bounds = cell_bounds(lo, hi, thresholds)
        excess = max(bounds['lower_residual_upper'], -bounds['upper_residual_lower'],
                     bounds['lower_trace_rate_upper'], bounds['upper_trace_rate_upper'])
        if excess <= float(tolerance):
            accepted += 1
            volume += F(1, 2**depth)
            deepest = max(deepest, depth)
            if worst is None:
                worst = bounds
            else:
                for key, value in bounds.items():
                    worst[key] = min(worst[key], value) if key.endswith('_lower') else max(worst[key], value)
        else:
            if depth >= max_depth or visited >= max_cells:
                raise ArithmeticError(f'Incomplete cover: depth={depth}, visited={visited}, excess={excess}')
            ll, lh, rl, rh = arithmetic.split(lo, hi, depth % 3)
            stack.extend([(rl, rh, depth+1), (ll, lh, depth+1)])
    if volume != 1:
        raise ArithmeticError('Exact covered volume is not one')
    return dict(cover_complete=True, exact_covered_volume=str(volume), accepted_cells=accepted,
                visited_cells=visited, maximum_depth=deepest, seconds=time.perf_counter()-start,
                polynomial_degrees=list(degrees), **worst)


def localization(candidate, trace_exp, bins=128):
    scalar, ends = arithmetic.scalar, arithmetic.ends
    d = candidate['dimension']
    blocks = validate(candidate)
    m = {k: scalar(v) for k, v in blocks[0]['model'].items()}
    k, e, Q, G, cap, R, T, gamma = [m[key] for key in
        ('kappa', 'drift_neighbor', 'variance', 'cross', 'cap', 'radius', 'horizon', 'gamma')]
    h = gamma-1
    ell, loading, risk = [scalar(max(F(b['model'][key]) for b in blocks))
                          for key in ('lambda_constant', 'lambda_self', 'risk_upper')]
    loading += scalar(max(abs(F(b['model']['lambda_neighbor'])) for b in blocks))
    shift = e*R+h*G*cap/iv.sqrt(scalar(d))
    if float((R-shift/k).a) <= 0:
        raise ArithmeticError('Exit threshold must be positive')
    prob = scalar(0)
    for j in range(1, bins+1):
        a = iv.exp(k*T*scalar(F(j-1, bins)))*(R-shift/k)+shift/k
        variance = Q/(2*k)*(iv.exp(2*k*T*scalar(F(j, bins)))-1)
        prob += 2*iv.exp(-a*a/(2*variance))
    prob *= d
    if ends(prob)[1] > 1:
        prob = scalar(1)
    weight = scalar(F(1, 20))
    denom = 2*weight*(k-e)-2*Q*weight*weight/d
    if float(denom.a) <= 0:
        raise ArithmeticError('Lyapunov domination must be positive')
    c0 = h*(gamma*risk/2*cap*cap+cap*ell)
    c1 = h*cap*loading
    linear = 2*weight*h*G*cap/iv.sqrt(scalar(d))+c1
    exponent = weight*Q+c0+linear*linear/(4*denom)
    continuation = iv.exp(exponent*T+weight*R*R)
    cp = h*(-m['risk_free_rate']+gamma*risk/2*cap*cap+cap*(ell+loading*R))
    cp = scalar(max(F(0), F(ends(cp)[1])))
    loc = iv.exp(cp*T)*(continuation+iv.exp(scalar(trace_exp)))*prob
    return dict(exit_probability_upper=ends(prob)[1], localization_upper=ends(loc)[1],
                quadratic_weight_exact='1/20', weight_generator_exponent_upper=ends(exponent)[1],
                continuation_upper_at_cube_boundary=ends(continuation)[1],
                coordinate_drift_shift_upper=ends(shift)[1], exit_partition_bins=bins)


def combine(candidate, covers):
    blocks = validate(candidate)
    d = candidate['dimension']
    if len(covers) != d or any(not c['cover_complete'] or c['exact_covered_volume'] != '1' for c in covers):
        raise ValueError('Every coordinate needs a complete verified cover')
    m = {k: F(v) for k, v in blocks[0]['model'].items()}
    lower = upper = -(m['gamma']-1)*m['risk_free_rate']*m['horizon']
    trace = F(0)
    for b, c in zip(blocks, covers):
        k = b['time_terms']
        eta_l = F(max(0., c['lower_residual_upper'], c['lower_trace_rate_upper']))
        eta_u = F(max(0., -c['upper_residual_lower'], c['upper_trace_rate_upper']))
        lower += (sum(map(F, b['lower']['coeff'][:k]))-eta_l)/d
        upper += (sum(map(F, b['upper']['coeff'][:k]))+eta_u)/d
        trace += F(c['base_factor_exponent_upper'])/d
    loc = localization(candidate, trace)
    low = iv.exp(arithmetic.scalar(lower))-arithmetic.scalar(F(loc['localization_upper']))
    high = iv.exp(arithmetic.scalar(upper))+arithmetic.scalar(F(loc['localization_upper']))
    if float(low.a) <= 0:
        raise ArithmeticError('Whole-space lower endpoint is nonpositive')
    loss = (1-iv.exp(iv.log(low/high)/arithmetic.scalar(m['gamma']-1)))*10000
    return dict(whole_space_certified=True, dimension=d, gamma_exact=str(m['gamma']),
                relative_ce_loss_upper_bp=arithmetic.ends(loss)[1],
                whole_space_optimal_factor_lower=arithmetic.ends(low)[0],
                whole_space_policy_factor_upper=arithmetic.ends(high)[1],
                local_cylinders_verified=d, maximum_local_verification_dimension=3,
                total_accepted_cells=sum(c['accepted_cells'] for c in covers),
                cover_verification_seconds=sum(c['seconds'] for c in covers),
                covariance_bound='diag(1-row_offsum) <= Sigma <= diag(1+row_offsum)',
                policy_is_covariance_metric_greedy=False,
                policy_updates_during_certification=0, **loc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions', nargs='+', type=int, default=[10, 20])
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--correlation', default='1/10')
    parser.add_argument('--tolerance', default='1/1000')
    parser.add_argument('--output-dir', default='results/highdim_network')
    args = parser.parse_args()
    iv.dps = 60
    out = ROOT/args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = {p.name: digest(p) for p in [Path(__file__), Path(arithmetic.__file__),
                                         Path(__file__).with_name('highdim_local_fit.py')]}
    for d in args.dimensions:
        path = out/f'candidate_d{d}.json'
        if args.fit:
            c = model(d, args.correlation)
            validate(c)
            train(c, epochs=args.epochs)
            c['fit_source_sha256'] = sources
            print(f'd={d} trained in {c["training"]["seconds"]:.2f}s', flush=True)
            import copy
            from highdim_local_fit import rounded
            reference = copy.deepcopy(c['blocks'][0])
            reference['base'] = rounded(np.mean([[float(F(x)) for x in b['base']] for b in c['blocks']], axis=0))
            reference['model'].update(lambda_constant='1/5', lambda_self='1/2', lambda_neighbor='1/4',
                                      risk_lower=str(min(F(b['model']['risk_lower']) for b in c['blocks'])),
                                      risk_upper=str(max(F(b['model']['risk_upper']) for b in c['blocks'])))
            fit_block(reference, fit_potentials=True)
            c['common_potential_fit_seconds'] = reference['fit']['seconds']
            potentials = {s: reference[s]['potential'] for s in ('lower', 'upper')}
            for i, b in enumerate(c['blocks']):
                fit_block(b, potentials=potentials)
                print(f'd={d} fitted {i+1}/{d}', flush=True)
            path.write_text(json.dumps(c, indent=2)+'\n', encoding='utf-8')
        c = json.loads(path.read_text(encoding='utf-8'))
        validate(c)
        covers = []
        for i, b in enumerate(c['blocks']):
            try:
                cover = verify_block(b, F(args.tolerance))
            except ArithmeticError as failure:
                if not args.fit:
                    raise
                if not str(failure).startswith('Incomplete cover:'):
                    raise
                fit_block(b, refinement=2, potentials={s: b[s]['potential'] for s in ('lower', 'upper')})
                path.write_text(json.dumps(c, indent=2)+'\n', encoding='utf-8')
                for attempt in range(8):
                    try:
                        cover = verify_block(b, F(args.tolerance))
                        break
                    except ArithmeticError as refined_failure:
                        if attempt == 7 or not str(refined_failure).startswith('Incomplete cover:'):
                            raise
                        shift = F(2**attempt, 8192)
                        b['lower']['coeff'][0] = str(F(b['lower']['coeff'][0])-shift)
                        b['upper']['coeff'][0] = str(F(b['upper']['coeff'][0])+shift)
                        b['dyadic_time_shift_exact'] = str(F(b.get('dyadic_time_shift_exact', '0'))+shift)
                        path.write_text(json.dumps(c, indent=2)+'\n', encoding='utf-8')
            covers.append(cover)
            print(f'd={d} verified {i+1}/{d}: {cover["accepted_cells"]} cells', flush=True)
        result = dict(**combine(c, covers), covers=covers, candidate_sha256=digest(path),
                      verifier_source_sha256=sources, tolerance_exact=str(F(args.tolerance)))
        for name, h in sources.items():
            assert digest(Path(__file__).with_name(name)) == h
        (out/f'certificate_d{d}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(f'd={d} CERTIFIED: {result["relative_ce_loss_upper_bp"]:.6f} bp', flush=True)


if __name__ == '__main__':
    main()
