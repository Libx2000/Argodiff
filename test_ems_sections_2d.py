"""Synthetic indexing/support fixtures only; no manuscript figures generated."""
import copy
import csv
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
import numpy as np
from ems_sections_2d import extract_sections,option,set_option,remap_project,check_original
from plot_ems_sections_2d import check_pair,edges

class Sections(unittest.TestCase):
    def setUp(self):
        self.lat=np.array([-10.,0.,10.,20.])
        self.lon=np.array([100.,110.,120.,130.,140.])
        self.depth=np.array([50.,10.,1.])
        self.pred=np.arange(120,dtype=np.float32).reshape(6,4,5)
        self.truth=self.pred+2
        self.valid=np.ones_like(self.truth,dtype=bool)
        self.valid[2,2,1]=False
    def extract(self,pred=None):
        return extract_sections(self.pred if pred is None else pred,self.truth,self.valid,
            self.lat,self.lon,self.depth,10,(110,140),130,(-10,20))
    def test_native_lines_and_channel_order(self):
        a=self.extract()
        np.testing.assert_array_equal(a['depth_m'],[1,10,50])
        np.testing.assert_array_equal(a['zonal_longitude'],[110,120,130,140])
        self.assertEqual(a['zonal_latitude'],10)
        self.assertEqual(a['zonal_prediction'].shape,(2,3,4))
        self.assertTrue(np.isnan(a['zonal_prediction'][0,0,0]))
        self.assertEqual(a['zonal_prediction'][0,0,1],self.pred[2,2,2])
        self.assertEqual(a['zonal_prediction'][1,0,1],self.pred[5,2,2])
        np.testing.assert_array_equal(a['meridional_prediction'][0,1],self.pred[1,:,3])
    def test_prediction_nan_not_silently_dropped(self):
        p=self.pred.copy();p[1,2,2]=np.nan
        with self.assertRaises(FloatingPointError):self.extract(p)
    def test_land_retained(self):
        a=self.extract();self.assertFalse(a['zonal_valid'][0,0,0])
        self.assertEqual(int(a['zonal_valid'].sum()),23)
    def test_pair_requires_identical_support(self):
        a=self.extract();a.update(date=np.asarray('TEST_ONLY'),lead=np.int32(30))
        b=copy.deepcopy(a)
        task=dict(phase='test',year=2019,scenario='nominal',season='winter',seed=1,members=50,steps=50)
        records=[dict(task=task),dict(task=copy.deepcopy(task))]
        check_pair([a,b],records)
        b['zonal_valid'][0,1,0]=False
        with self.assertRaises(AssertionError):check_pair([a,b],records)
    def test_pair_rejects_different_seed(self):
        a=self.extract();a.update(date=np.asarray('TEST_ONLY'),lead=np.int32(30))
        task=dict(phase='test',year=2019,scenario='nominal',season='winter',seed=1,members=50,steps=50)
        other=dict(task,seed=2)
        with self.assertRaises(ValueError):check_pair([a,copy.deepcopy(a)],[dict(task=task),dict(task=other)])
    def test_edges_and_invalid_line(self):
        np.testing.assert_array_equal(edges([1,3,5]),[0,2,4,6])
        with self.assertRaises(ValueError):
            extract_sections(self.pred,self.truth,self.valid,self.lat,self.lon,self.depth,90,(110,140),130,(-10,20))
    def test_option_rewrite_does_not_change_sampling(self):
        argv=['--days','90','--members','50','--sampling-steps','50','--output-dir','old']
        set_option(argv,'--output-dir','new')
        self.assertEqual(option(argv,'--members'),'50')
        self.assertEqual(option(argv,'--sampling-steps'),'50')
        self.assertEqual(option(argv,'--output-dir'),'new')
        self.assertTrue(str(remap_project('/fs6/home/x/diffArgo/configs/a.yaml')).endswith(str(Path('configs/a.yaml'))))
    def test_archived_global_score_guard(self):
        ds=SimpleNamespace(levels=3,archive=SimpleNamespace(read_state=lambda _: (self.truth,self.valid)),
            latitude_weights=np.ones((1,4,1),np.float32))
        with tempfile.TemporaryDirectory(prefix='ems_section_test_only_') as tmp:
            case=Path(tmp)
            records=[dict(lead=2,domain='glorys',variable=var,depth_index=-1,
                count=int(self.valid[v*3:(v+1)*3].sum()),rmse=2.)
                for v,var in enumerate(('temperature','salinity'))]
            with (case/'paper_grid_metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
            _,_,checks=check_original(case,ds,'TEST_ONLY',2,self.pred,np.stack([self.pred,self.pred]))
            self.assertTrue(checks['passed'])
            altered=self.pred.copy();altered[1,1,1]+=1
            with self.assertRaises(RuntimeError):
                check_original(case,ds,'TEST_ONLY',2,altered,np.stack([altered,altered]))

if __name__=='__main__':unittest.main(verbosity=2)
