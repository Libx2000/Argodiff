"""Civil environmental sensitivity only; no invented platform or safety claim."""
from pathlib import Path
import csv, json
import numpy as np
from oe_joint_diagnostics import physics_fields
from oe_joint_calibration import calibrated_members,lead_factors
from paper_metrics import score_samples


def neutral_volume_ml_per_kg(density):
    density = np.asarray(density, float)
    return np.where(np.isfinite(density) & (density > 0), 1e6/density, np.nan)


def rigid_device_response(estimated_density, actual_density, mass_kg, body_volume_m3,
                          min_command_m3, max_command_m3, gravity=9.81):
    """Quasi-static hypothetical rigid-body calculation, NOT a validated controller.

    Caller must supply all device values and provenance. No fabricated defaults.
    """
    values = [mass_kg, body_volume_m3, min_command_m3, max_command_m3]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Real or explicitly hypothetical device parameters must be supplied')
    if mass_kg <= 0 or body_volume_m3 <= 0 or min_command_m3 >= max_command_m3:
        raise ValueError('Invalid rigid-body/actuator parameters')
    command = mass_kg/np.asarray(estimated_density)-body_volume_m3
    applied = np.clip(command, min_command_m3, max_command_m3)
    return dict(command_m3=command, applied_m3=applied,
                saturated=(command < min_command_m3) | (command > max_command_m3),
                residual_buoyancy_N=gravity*(np.asarray(actual_density)*(body_volume_m3+applied)-mass_kg))


class BuoyancyObserver:
    def __init__(self, config, output, candidates=None):
        self.c, self.out = config, Path(output)
        self.device = config.get('device', {})
        self.candidates=candidates or [dict(id='raw',kind='raw')]

    def __call__(self, dataset, target, lead, center, ensemble, deterministic=False):
        points = self.c['snapshot_points']
        ii = [int(np.argmin(abs(dataset.lat-p['latitude']))) for p in points]
        jj = [int(np.argmin(abs((dataset.lon-p['longitude']+180)%360-180))) for p in points]
        raw, mask = dataset.archive.read_state(target)
        truth = np.where(mask, raw, np.nan)[:,ii,jj]
        depth = np.abs(dataset.archive.coordinates()['depth'])
        lat, lon = dataset.lat[ii], dataset.lon[jj]
        if deterministic:
            ensemble = ensemble[:1]
        factors=lead_factors(self.c['reference_calibration'],lead,center.shape[0])
        candidates=[dict(id='raw',kind='raw')] if deterministic else self.candidates
        cf, _, _ = physics_fields(center[None,:,ii,jj], depth, lat, lon)
        tf, d, _ = physics_fields(truth[None], depth, lat, lon)
        vref=neutral_volume_ml_per_kg(tf['rho'][0])
        vcenter = neutral_volume_ml_per_kg(cf['rho'][0])
        rows = []
        for candidate,en,_ in calibrated_members(ensemble[:,:,ii,jj],center[:,ii,jj],depth,lat,lon,
                factors,candidates,self.c['regions'],self.c['regional_feather_degrees']):
            ef,_,_=physics_fields(en,depth,lat,lon)
            v=neutral_volume_ml_per_kg(ef['rho'])
            for j in range(len(points)):
                for k, z in enumerate(d):
                    members = v[:,k,j]
                    if not (np.isfinite(members).all() and np.isfinite(vref[k,j]) and np.isfinite(vcenter[k,j])):
                        continue
                    q = np.quantile(members, [.05,.95])
                    stats = score_samples(np.array([vcenter[k,j]]), np.array([vref[k,j]]), members[:,None])
                    rows.append(dict(date=str(target),lead=lead,variant=candidate['id'],site=j+1,latitude=float(lat[j]),longitude=float(lon[j]),
                                     depth_m=float(z),members=len(members),reference_ml_kg=float(vref[k,j]),
                                     center_ml_kg=float(vcenter[k,j]),p05_ml_kg=float(q[0]),p95_ml_kg=float(q[1]),
                                     center_error_ml_kg=float(vcenter[k,j]-vref[k,j]),crps_ml_kg=stats['crps'],
                                     coverage90=stats['coverage90'],scope='normalized_environment_sensitivity'))
        if rows:
            path=self.out/'buoyancy_sensitivity.csv';exists=path.exists()
            with path.open('a',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                if not exists:writer.writeheader()
                writer.writerows(rows)
        if lead in self.c['snapshot_leads']:
            # Actual fixed-latitude longitude-depth transect, never interpolation
            # between the three isolated profile sites.
            i=int(np.argmin(abs(dataset.lat-self.c['transect']['latitude'])))
            lo,hi=self.c['transect']['longitude'];js=np.flatnonzero((dataset.lon>=lo)&(dataset.lon<=hi))
            if not len(js):raise ValueError('No grid points in predefined transect')
            section=dict(depth_m=depth,latitude=dataset.lat[i],longitude=dataset.lon[js],lead=np.asarray(lead),
                center=center[:,i,js],truth=np.where(mask,raw,np.nan)[:,i,js],ensemble=ensemble[:,:,i,js],
                ensemble_role=np.asarray('ensemble is raw; explicitly named ensemble_* store each scored variant'))
            for candidate,en,_ in calibrated_members(ensemble[:,:,i,js],center[:,i,js],depth,
                    np.full(len(js),dataset.lat[i]),dataset.lon[js],factors,candidates,self.c['regions'],self.c['regional_feather_degrees']):
                section['ensemble_'+candidate['id']]=en.astype(np.float32)
            np.savez_compressed(self.out/f'vertical_section_day{lead:03d}.npz',**section)
        status=dict(mode='normalized environmental sensitivity; no actual device supplied',
                    quantity='1e6/rho, mL/kg required total neutral-displacement volume',
                    independent_engineering_validation=False,device_parameters_confirmed=False,
                    not_claimed=['measured vehicle response','controller validation','operational safety','energy saving'],
                    interpretation='This is a nonlinear transformation of density skill, not independent evidence of device benefit.')
        (self.out/'buoyancy_scope.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
