"""Bounded soft GSW stability gradients, without truth or a Kalman update.

The gradient is a local central finite-difference gradient of GSW.Nsquared,
including SA_from_SP and CT_from_pt. It is NOT backpropagation through the
denoiser and is not an exact posterior sampler. Only adjacent levels interact.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import gsw


DEFAULTS = dict(denoise=False, center=False, n2_floor=-1e-6, n2_scale=1e-4,
                fd_temperature=1e-3, fd_salinity=1e-3, start=.8, every=10,
                center_iterations=6, denoise_iterations=1,
                temperature_step=.04, salinity_step=.004,
                max_temperature_change=.20, max_salinity_change=.02,
                anchor_weight=.01, observation_weight=.01,
                max_observation_cost_increase=.05,
                conserve_sampled_column=True, backtracks=8)


class StabilityOperator:
    def __init__(self, depth, latitude, longitude, options=None):
        self.c = dict(DEFAULTS, **(options or {}))
        d = np.abs(np.asarray(depth, float))
        if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
            raise ValueError('At least two finite depth levels required')
        self.order = np.argsort(d)
        self.depth = d[self.order]
        if (np.diff(self.depth) <= 0).any():
            raise ValueError('Depth levels must be distinct')
        self.levels = len(d)
        self.channels = np.r_[self.order, self.order + self.levels]
        self.inverse = np.argsort(self.channels)
        la, lo = np.meshgrid(np.asarray(latitude, float), np.asarray(longitude, float), indexing='ij')
        if not np.isfinite(la).all() or (abs(la) > 90).any() or not np.isfinite(lo).all():
            raise ValueError('Invalid grid positions')
        self.shape = la.shape
        self.lat, self.lon = la.reshape(1, 1, -1), np.mod(lo, 360).reshape(1, 1, -1)
        self.p = gsw.p_from_z(-self.depth[None, :, None], self.lat)
        # GSW's salinity-anomaly atlas is undefined at some land/coastal
        # coordinates. This fixed geometric support is not forecast filtering.
        self.geometry = np.isfinite(gsw.SA_from_SP(np.full_like(self.p,35.),self.p,self.lon,self.lat))
        self.weight = np.cos(np.deg2rad(self.lat))
        dz = np.diff(self.depth)
        self.column_weights = np.r_[dz[0]/2, (dz[:-1]+dz[1:])/2, dz[-1]/2].reshape(1, -1, 1)
        self.records = []
        for key in ('fd_temperature', 'fd_salinity', 'n2_scale', 'temperature_step',
                    'salinity_step', 'max_temperature_change', 'max_salinity_change'):
            if not np.isfinite(self.c[key]) or self.c[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if not 0 <= self.c['start'] < 1 or int(self.c['every']) < 1:
            raise ValueError('Invalid denoising schedule')
        if self.c['n2_floor'] > 0:
            raise ValueError('This experiment allows weak negatives; do not impose a positive floor')

    def _ordered(self, x):
        x = np.asarray(x, float)
        if x.ndim != 4 or x.shape[1:] != (2*self.levels, *self.shape):
            raise ValueError('Expected batch,2*depth,latitude,longitude')
        return x[:, self.channels].reshape(len(x), 2*self.levels, -1)

    def _n2(self, x, pairs=None):
        l = self.levels
        # Computational extension at search boundaries, not a replacement for
        # final ocean-mask diagnostics. The solver separately enforces T/S bounds.
        t, s = np.clip(x[:, :l], -3, 40), np.clip(x[:, l:], 0, 42)
        sa = gsw.SA_from_SP(s, self.p, self.lon, self.lat)
        ct = gsw.CT_from_pt(sa, t)
        n2 = gsw.Nsquared(sa, ct, self.p, lat=self.lat, axis=1)[0]
        if pairs is not None and not np.isfinite(n2[pairs]).all():
            raise FloatingPointError('GSW produced nonfinite stability on computationally valid inputs')
        return np.where(np.isfinite(n2),n2,0.)

    def _prepare(self, state, mask):
        x = self._ordered(state)
        mask = np.broadcast_to(np.asarray(mask, bool), np.shape(state))
        m = self._ordered(mask).astype(bool)
        if not np.isfinite(x[m]).all():
            raise FloatingPointError('Nonfinite state on declared ocean support')
        l = self.levels
        safe = np.where(m, x, np.r_[np.full(l, 10.), np.full(l, 35.)][None, :, None])
        valid = m[:, :l] & m[:, l:] & self.geometry
        pairs = valid[:, :-1] & valid[:, 1:]
        return safe, m, pairs

    def _loss(self, x, pairs):
        n2 = self._n2(x,pairs)
        w = pairs * self.weight
        denominator = w.sum()
        violation = np.maximum(self.c['n2_floor'] - n2, 0) / self.c['n2_scale']
        loss = float((violation**2*w).sum()/denominator) if denominator else 0.
        derivative = -2*violation*w/(max(denominator, 1e-30)*self.c['n2_scale'])
        return loss, derivative, n2

    def loss_gradient(self, state, mask):
        """Return actual soft loss and its FD gradient in original channel order."""
        x, m, pairs = self._prepare(state, mask)
        loss, dn, _ = self._loss(x, pairs)
        gradient = np.zeros_like(x)
        if loss:
            l = self.levels
            upper = np.arange(l-1)
            for offset, epsilon in ((0, self.c['fd_temperature']), (l, self.c['fd_salinity'])):
                for parity in (0, 1):
                    selected = np.arange(parity, l, 2) + offset
                    plus, minus = x.copy(), x.copy()
                    plus[:, selected] += epsilon
                    minus[:, selected] -= epsilon
                    jac = (self._n2(plus,pairs)-self._n2(minus,pairs))/(2*epsilon)
                    gradient[:, offset:offset+l-1] += dn*jac*(upper % 2 == parity)[None, :, None]
                    gradient[:, offset+1:offset+l] += dn*jac*((upper+1) % 2 == parity)[None, :, None]
            gradient *= m
        gradient = gradient[:, self.inverse].reshape(np.shape(state))
        return loss, gradient

    def _project(self, delta, original, m):
        l = self.levels
        for sl, cap, lower, upper in ((slice(0,l), self.c['max_temperature_change'], -3., 40.),
                                      (slice(l,2*l), self.c['max_salinity_change'], 0., 42.)):
            u, valid, base = delta[:, sl], m[:, sl] & self.geometry, original[:, sl]
            u *= valid
            if self.c['conserve_sampled_column']:
                w = self.column_weights*valid
                mean = (u*w).sum(1, keepdims=True)/np.maximum(w.sum(1, keepdims=True), 1e-30)
                u -= mean*valid
            ratio = np.minimum(1., cap/np.maximum(abs(u).max(1, keepdims=True), 1e-30))
            allowed = np.where(u > 0, (upper-base)/np.maximum(u,1e-30),
                               np.where(u < 0, (base-lower)/np.maximum(-u,1e-30), 1.))
            allowed = np.where(valid, allowed, 1.)
            ratio = np.minimum(ratio, np.clip(allowed.min(1, keepdims=True), 0., 1.))
            delta[:, sl] = u*ratio
        return delta

    def correct_numpy(self, state, mask, iterations, observation=None, observation_mask=None, role='center'):
        shape = np.shape(state)
        original, m, pairs = self._prepare(state, mask)
        current = original.copy()
        initial, _, before_n2 = self._loss(current, pairs)
        l = self.levels
        unit = np.r_[np.ones(l), np.full(l,.1)].reshape(1,-1,1)
        step = np.r_[np.full(l,self.c['temperature_step']), np.full(l,self.c['salinity_step'])].reshape(1,-1,1)
        if observation is None:
            observed, om = original.copy(), np.zeros_like(m)
        else:
            observed = self._ordered(np.broadcast_to(np.asarray(observation), shape))
            om = self._ordered(np.broadcast_to(np.asarray(observation_mask, bool), shape)).astype(bool) & m
            if not np.isfinite(observed[om]).all():
                raise FloatingPointError('Nonfinite supplied observation on its valid mask')
            observed = np.where(om, observed, original)
        noise = np.r_[np.full(l,.5), np.full(l,.1)].reshape(1,-1,1)
        obs_cost = lambda q: float((((q-observed)/noise)**2*om).sum()/max(om.sum(),1))
        anchor = lambda q: float((((q-original)/unit)**2*m).sum()/max(m.sum(),1))
        baseline_obs = obs_cost(original)
        accepted = 0
        for _ in range(int(iterations)):
            physical_loss, _, _ = self._loss(current, pairs)
            if physical_loss == 0:
                break
            as_grid = current[:, self.inverse].reshape(shape)
            _, gradient_grid = self.loss_gradient(as_grid, mask)
            grad = self._ordered(gradient_grid)
            grad += 2*self.c['anchor_weight']*(current-original)/unit**2*m/max(m.sum(),1)
            grad += 2*self.c['observation_weight']*(current-observed)/noise**2*om/max(om.sum(),1)
            direction = -grad*unit**2
            for sl in (slice(0,l), slice(l,2*l)):
                direction[:,sl] /= np.maximum(abs(direction[:,sl]).max(1,keepdims=True), 1e-30)
            direction *= step
            old_obj = physical_loss+self.c['anchor_weight']*anchor(current)+self.c['observation_weight']*obs_cost(current)
            for k in range(int(self.c['backtracks'])):
                delta = self._project(current-original+direction*(.5**k), original, m)
                trial = original+delta
                new_loss, _, _ = self._loss(trial, pairs)
                cost = obs_cost(trial)
                obj = new_loss+self.c['anchor_weight']*anchor(trial)+self.c['observation_weight']*cost
                if (obj < old_obj-1e-14 and new_loss <= physical_loss+1e-14
                        and cost <= baseline_obs*(1+self.c['max_observation_cost_increase'])+1e-6):
                    current = trial
                    accepted += 1
                    break
            else:
                break
        final, _, after_n2 = self._loss(current, pairs)
        delta = current-original
        w = self.column_weights
        budget_t = abs((delta[:,:l]*w*m[:,:l]).sum(1)).max(initial=0)
        budget_s = abs((delta[:,l:]*w*m[:,l:]).sum(1)).max(initial=0)
        event = lambda n: float(((n < -1e-5)*pairs*self.weight).sum()/max((pairs*self.weight).sum(),1e-30))
        self.records.append(dict(role=role, initial_loss=initial, final_loss=final, accepted_steps=accepted,
                                 max_temperature_delta=float(abs(delta[:,:l]).max(initial=0)),
                                 max_salinity_delta=float(abs(delta[:,l:]).max(initial=0)),
                                 sampled_column_temperature_integral_delta_max=float(budget_t),
                                 sampled_column_salinity_integral_delta_max=float(budget_s),
                                 obs_cost_before=baseline_obs, obs_cost_after=obs_cost(current),
                                 severe_event_before=event(before_n2), severe_event_after=event(after_n2),
                                 valid_layer_pairs=int(pairs.sum()),
                                 excluded_gsw_geometry_pairs=int(((m[:,:l-1]&m[:,1:l]&m[:,l:-1]&m[:,l+1:])&~pairs).sum()),
                                 cast_error_max=0.,
                                 actual_loss_after_cast=final, residual_box_reverted=0, post_box_loss=final))
        output = current[:, self.inverse].reshape(shape)
        return np.where(np.broadcast_to(mask,shape), output, state)

    def correct_tensor(self, state, mask, iterations, observation=None, observation_mask=None, role='center'):
        import torch
        array = lambda v: None if v is None else v.detach().cpu().numpy()
        result = self.correct_numpy(array(state), array(mask).astype(bool), iterations,
                                    array(observation), array(observation_mask), role)
        output=torch.as_tensor(result, device=state.device, dtype=state.dtype)
        actual=output.detach().cpu().numpy().astype(float)
        valid=np.broadcast_to(array(mask).astype(bool),result.shape)
        self.records[-1]['cast_error_max']=float(abs(actual[valid]-result[valid]).max(initial=0))
        safe,_,pairs=self._prepare(actual,valid)
        self.records[-1]['actual_loss_after_cast']=self._loss(safe,pairs)[0]
        return output

    def denoise(self, clean, background, mean, std, residual_scale, mask, observation, observation_mask, step, progress, final):
        if (not self.c['denoise'] or residual_scale <= 0 or progress < self.c['start']
                or (step % self.c['every'] and not final)):
            return clean
        import torch
        physical = background+residual_scale*(clean*std+mean)
        corrected = self.correct_tensor(physical, mask, self.c['denoise_iterations'], observation, observation_mask, 'denoise')
        normalized = ((corrected-background)/residual_scale-mean)/std
        proposal=torch.where(mask.bool(), normalized, clean).detach().clamp(-8,8)
        actual=background+residual_scale*(proposal*std+mean)
        safe,_,pairs=self._prepare(actual.detach().cpu().numpy(),mask.detach().cpu().numpy().astype(bool))
        actual_loss=self._loss(safe,pairs)[0]
        before=self.records[-1]['initial_loss']
        if actual_loss>before+1e-10:
            self.records[-1]['residual_box_reverted']=1
            self.records[-1]['post_box_loss']=before
            return clean
        self.records[-1]['post_box_loss']=actual_loss
        return proposal

    def write_day(self, output, lead, date):
        if not self.records:
            return
        path = Path(output)/'stability_solver.csv'
        exists = path.exists()
        with path.open('a', encoding='utf-8', newline='') as f:
            rows = [dict(date=str(date),lead=lead,**r) for r in self.records]
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        self.records.clear()
