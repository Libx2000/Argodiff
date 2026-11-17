"""Tiny real-U-Net/NetCDF training smoke; no ocean archive or GPU required."""
import math,sys,tempfile,unittest
from pathlib import Path
from datetime import date,timedelta
from unittest.mock import patch
import numpy as np
import netCDF4
import torch
import yaml
from diffargo.model import build_model
from train_residual_rollout import main


class EndToEndTest(unittest.TestCase):
    def test_train_validate_and_save(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'configs').mkdir();data=root/'data'
            (data/'T').mkdir(parents=True);(data/'S').mkdir()
            dates={2000:[date(2000,1,1)+timedelta(days=i) for i in range(10)],
                   2018:[date(2018,12,30),date(2018,12,31)],2019:[]}
            for month in [1,4,7,10]:
                anchor=date(2019,month,1)
                dates[2019]+=[anchor+timedelta(days=i) for i in range(-2,3) if (anchor+timedelta(days=i)).year==2019]
            lat=np.linspace(-60,60,8).astype('f4');lon=(np.arange(8)*45).astype('f4')
            for year,days in dates.items():
                for kind,var in [('T','t'),('S','s')]:
                    with netCDF4.Dataset(data/kind/f'{kind}_{year}.nc','w') as ds:
                        for name,n in [('time',len(days)),('depth',1),('lat',8),('lon',8)]:ds.createDimension(name,n)
                        tv=ds.createVariable('time','f8',('time',));tv.units='days since 1900-01-01 00:00:00'
                        tv[:]=[(d-date(1900,1,1)).days for d in days]
                        ds.createVariable('depth','f4',('depth',))[:]=[10]
                        ds.createVariable('lat','f4',('lat',))[:]=lat;ds.createVariable('lon','f4',('lon',))[:]=lon
                        values=np.stack([np.full((1,8,8),(10 if kind=='T' else 35)+.01*math.sin(d.toordinal()),dtype='f4') for d in days])
                        values[:,:,0,0]=np.nan
                        ds.createVariable(var,'f4',('time','depth','lat','lon'))[:]=values
            stats=root/'stats.npz';mask=np.ones((1,8,8),np.float32);mask[:,0,0]=0
            np.savez_compressed(stats,state_mean=np.array([10,35],np.float32),state_std=np.array([5,1],np.float32),
                increment_mean=np.zeros(2,np.float32),increment_std=np.array([.1,.05],np.float32),static_mask=mask)
            residual_stats=root/'res.npz'
            np.savez_compressed(residual_stats,residual_mean=np.zeros(2,np.float32),residual_std=np.array([.05,.02],np.float32))
            cfg={'seed':7,'data':{'root':str(data),'temperature_subdir':'T','salinity_subdir':'S',
                'temperature_pattern':'T_{year}.nc','salinity_pattern':'S_{year}.nc','temperature_variable':'t','salinity_variable':'s',
                'depth_name':'depth','latitude_name':'lat','longitude_name':'lon','time_name':'time','train_years':[2000,2000],
                'val_years':[2019,2019],'test_years':[2019,2019],'context_days':3,'lead_days':1,'stats_path':str(stats),'file_cache_size':4},
                'model':{'base_channels':4,'channel_multipliers':[1,2],'time_embedding_dim':16,'dropout':0.,'attention_at_bottleneck':False},
                'diffusion':{'timesteps':16,'beta_schedule':'cosine'},'train':{'output_dir':'unused','ema_decay':.99}}
            config=root/'configs/test.yaml';config.write_text(yaml.safe_dump(cfg))
            model=build_model(cfg,2,12)
            with torch.no_grad():model.output_conv.conv.bias.zero_()
            source={k:v.clone() for k,v in model.state_dict().items()}
            det=root/'det.pt';res=root/'res.pt'
            torch.save({'model':source},det);torch.save({'model':source,'objective':'residual_diffusion'},res)
            output=root/'run'
            args=['trainer','--smoke','--config',str(config),'--deterministic-checkpoint',str(det),
                '--source-checkpoint',str(res),'--residual-stats',str(residual_stats),'--output-dir',str(output)]
            with patch.object(sys,'argv',args),patch('torch.cuda.is_available',return_value=False):main()
            self.assertTrue((output/'checkpoint_last.pt').is_file())
            payload=torch.load(output/'checkpoint_best.pt',map_location='cpu',weights_only=False)
            self.assertEqual(payload['objective'],'residual_diffusion')
            self.assertTrue(math.isfinite(payload['validation']['score']))
            self.assertTrue(any(not torch.equal(v,source[k]) for k,v in payload['model'].items()))
            self.assertTrue((output/'training_metrics.csv').is_file())
            # Exercise the actual cycling entry point: with no Argo, the paired
            # observation-only center must reproduce the deterministic baseline.
            from argo_guided_cycle import main as cycle_main
            common=['cycle','--config',str(config),'--deterministic-checkpoint',str(det),
                '--residual-checkpoint',str(output/'checkpoint_best.pt'),'--residual-stats',str(residual_stats),
                '--start','2019-01-01','--days','2','--members','2','--sampling-steps','4',
                '--sampling-eta','0','--residual-scale','1','--save-daily','compact']
            for mode,extra in [('baseline',['--deterministic-only']),
                               ('free0',['--argo-stop-after','0','--argo-root',str(root/'missing'),
                                         '--require-argo','--member-mean-cycle','--individual-member-guidance','--paired-guidance-center']),
                               ('paired',['--member-mean-cycle','--individual-member-guidance','--paired-guidance-center'])]:
                args=common+extra+['--output-dir',str(root/mode)]
                with patch.object(sys,'argv',args),patch('torch.cuda.is_available',return_value=False):cycle_main()
            for day in ['2019-01-02','2019-01-03']:
                with np.load(root/'free0'/f'{day}.npz') as free, np.load(root/'baseline'/f'{day}.npz') as base:
                    np.testing.assert_allclose(free['analysis'],base['analysis'],atol=1e-7,equal_nan=True)
                    self.assertEqual(float(np.nanmax(free['analysis_spread'])),0.)
                with np.load(root/'baseline'/f'{day}.npz',allow_pickle=False) as b,\
                     np.load(root/'paired'/f'{day}.npz',allow_pickle=False) as c:
                    np.testing.assert_allclose(c['analysis'],b['analysis'],rtol=0,atol=1e-6,equal_nan=True)


if __name__=='__main__':unittest.main()
