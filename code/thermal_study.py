"""Controlled EnergyPlus load-sensitivity experiment with prescribed solar gains.

No claim of stock or field validation: a common test cell isolates the propagation
of each receiver's shading-fraction error. Native shortwave weather inputs are zero;
identical diffuse gain and altered direct gain are explicitly imposed.
"""
from __future__ import annotations
import json,subprocess,time,argparse,csv
from pathlib import Path
import numpy as np
import pandas as pd
import revision_core as c
ROOT=c.ROOT
RESULTS=ROOT/'final_results'
EP=next((ROOT/'dependencies').glob('EnergyPlus-*/energyplus'))
WEATHER=ROOT/'data/rotterdam_pvgis_tmy.epw'

def thermal_weather():
    """Keep all meteorology; move shortwave input entirely to prescribed gains."""
    out=ROOT/'data/rotterdam_thermal_driver_zero_shortwave.epw'
    if not out.exists():
        lines=WEATHER.read_text().splitlines();data=list(csv.reader(lines[8:]))
        for row in data:
            # EPW global/direct/diffuse shortwave and their photometric fields.
            for col in range(13,20):row[col]='0'
        with out.open('w',newline='') as f:
            f.write('\n'.join(lines[:8])+'\n');csv.writer(f,lineterminator='\n').writerows(data)
    return out

def model(schedule_path,free_float=False,mass='heavy',steps=6):
    m={}
    def add(typ,name,**kw):m.setdefault(typ,{})[name]=kw
    add('Version','Version',version_identifier='26.1')
    add('Timestep','Timestep',number_of_timesteps_per_hour=steps)
    add('SimulationControl','Control',do_zone_sizing_calculation='No',do_system_sizing_calculation='No',do_plant_sizing_calculation='No',run_simulation_for_sizing_periods='No',run_simulation_for_weather_file_run_periods='Yes')
    add('Building','Controlled test cell',north_axis=0,terrain='City',solar_distribution='MinimalShadowing',maximum_number_of_warmup_days=50,minimum_number_of_warmup_days=6,loads_convergence_tolerance_value=.04,temperature_convergence_tolerance_value=.2)
    add('RunPeriod','Annual',begin_month=1,begin_day_of_month=1,begin_year=2021,end_month=12,end_day_of_month=31,end_year=2021,use_weather_file_daylight_saving_period='No',use_weather_file_holidays_and_special_days='No')
    add('GlobalGeometryRules','Geometry',starting_vertex_position='UpperLeftCorner',vertex_entry_direction='Counterclockwise',coordinate_system='World')
    add('HeatBalanceAlgorithm','Heat balance',algorithm='ConductionTransferFunction')
    add('SurfaceConvectionAlgorithm:Inside','Inside',algorithm='TARP')
    add('SurfaceConvectionAlgorithm:Outside','Outside',algorithm='DOE-2')
    add('Zone','Cell',ceiling_height=3,volume=75,floor_area=25)
    for name,th,k,rho,cp in [('Insulation',.1,.04,35,1400),('Concrete',.15,1.4,2200,880),('Gypsum',.0125,.16,800,1090)]:
        add('Material',name,roughness='MediumRough',thickness=th,conductivity=k,density=rho,specific_heat=cp,thermal_absorptance=.9,solar_absorptance=.6,visible_absorptance=.6)
    add('Construction','Envelope',outside_layer='Insulation',layer_2='Concrete' if mass=='heavy' else 'Gypsum')
    add('WindowMaterial:SimpleGlazingSystem','Glazing',u_factor=1.6,solar_heat_gain_coefficient=.5,visible_transmittance=.6)
    add('Construction','Window',outside_layer='Glazing')
    surfaces={
      'South':('Wall',[(0,0,3),(0,0,0),(5,0,0),(5,0,3)]),
      'East':('Wall',[(5,0,3),(5,0,0),(5,5,0),(5,5,3)]),
      'North':('Wall',[(5,5,3),(5,5,0),(0,5,0),(0,5,3)]),
      'West':('Wall',[(0,5,3),(0,5,0),(0,0,0),(0,0,3)]),
      'Roof':('Roof',[(0,5,3),(0,0,3),(5,0,3),(5,5,3)]),
      'Floor':('Floor',[(0,0,0),(0,5,0),(5,5,0),(5,0,0)])}
    for name,(typ,pts) in surfaces.items():
        add('BuildingSurface:Detailed',name,surface_type=typ,construction_name='Envelope',zone_name='Cell',outside_boundary_condition='Adiabatic' if typ=='Floor' else 'Outdoors',sun_exposure='NoSun' if typ=='Floor' else 'SunExposed',wind_exposure='NoWind' if typ=='Floor' else 'WindExposed',number_of_vertices=4,vertices=[dict(vertex_x_coordinate=x,vertex_y_coordinate=y,vertex_z_coordinate=z) for x,y,z in pts])
    window={f'vertex_{i}_{axis}_coordinate':value for i,pt in enumerate([(1,0,2.5),(1,0,.5),(4,0,.5),(4,0,2.5)],1) for axis,value in zip('xyz',pt)}
    add('FenestrationSurface:Detailed','Equivalent glazing',surface_type='Window',construction_name='Window',building_surface_name='South',number_of_vertices=4,**window)
    add('ScheduleTypeLimits','Any',numeric_type='Continuous')
    add('Schedule:Constant','Always',schedule_type_limits_name='Any',hourly_value=1)
    add('Schedule:Constant','Heating',schedule_type_limits_name='Any',hourly_value=20)
    add('Schedule:Constant','Cooling',schedule_type_limits_name='Any',hourly_value=26)
    add('Schedule:Constant','Dual control',schedule_type_limits_name='Any',hourly_value=4)
    add('Schedule:File','Prescribed solar',schedule_type_limits_name='Any',file_name=str(Path(schedule_path).resolve()),column_number=1,rows_to_skip_at_top=0,number_of_hours_of_data=8760,column_separator='Comma',interpolate_to_timestep='No',minutes_per_item=60,adjust_schedule_for_daylight_savings='No')
    add('OtherEquipment','Prescribed solar gains',fuel_type='None',zone_or_zonelist_or_space_or_spacelist_name='Cell',schedule_name='Prescribed solar',design_level_calculation_method='EquipmentLevel',design_level=1,fraction_latent=0,fraction_radiant=.7,fraction_lost=0)
    add('OtherEquipment','Internal gains',fuel_type='None',zone_or_zonelist_or_space_or_spacelist_name='Cell',schedule_name='Always',design_level_calculation_method='EquipmentLevel',design_level=100,fraction_latent=0,fraction_radiant=.3,fraction_lost=0)
    add('ZoneInfiltration:DesignFlowRate','Infiltration',zone_or_zonelist_or_space_or_spacelist_name='Cell',schedule_name='Always',design_flow_rate_calculation_method='AirChanges/Hour',air_changes_per_hour=.5,constant_term_coefficient=1)
    if not free_float:
        add('ThermostatSetpoint:DualSetpoint','Dual',heating_setpoint_temperature_schedule_name='Heating',cooling_setpoint_temperature_schedule_name='Cooling')
        add('ZoneControl:Thermostat','Thermostat',zone_or_zonelist_name='Cell',control_type_schedule_name='Dual control',control_1_object_type='ThermostatSetpoint:DualSetpoint',control_1_name='Dual')
        add('ZoneHVAC:IdealLoadsAirSystem','Ideal loads',availability_schedule_name='Always',zone_supply_air_node_name='Supply',heating_limit='NoLimit',cooling_limit='NoLimit',dehumidification_control_type='None',humidification_control_type='None',outdoor_air_economizer_type='NoEconomizer',heat_recovery_type='None')
        add('ZoneHVAC:EquipmentConnections','Connections',zone_name='Cell',zone_conditioning_equipment_list_name='Equipment',zone_air_inlet_node_or_nodelist_name='Supply',zone_air_node_name='Zone air',zone_return_air_node_or_nodelist_name='Return')
        add('ZoneHVAC:EquipmentList','Equipment',equipment=[dict(zone_equipment_object_type='ZoneHVAC:IdealLoadsAirSystem',zone_equipment_name='Ideal loads',zone_equipment_cooling_sequence=1,zone_equipment_heating_or_no_load_sequence=1)])
    variables=['Zone Mean Air Temperature','Zone Operative Temperature','Zone Ideal Loads Zone Sensible Heating Energy','Zone Ideal Loads Zone Sensible Cooling Energy','Zone Ideal Loads Zone Sensible Heating Rate','Zone Ideal Loads Zone Sensible Cooling Rate','Zone Other Equipment Total Heating Energy','Surface Window Transmitted Solar Radiation Energy']
    if free_float:variables=[v for v in variables if 'Ideal Loads' not in v]
    for i,name in enumerate(variables):add('Output:Variable',str(i),key_value='*',variable_name=name,reporting_frequency='Hourly')
    add('OutputControl:Files','Files',output_csv='Yes',output_eso='No',output_mtr='No')
    return m

def run_case(directory,gains,free_float=False,mass='heavy',steps=6):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    report=directory/'summary.json'
    if report.exists():return json.loads(report.read_text())
    schedule=directory/'gains.csv';np.savetxt(schedule,gains,fmt='%.8f')
    inp=directory/'cell.epJSON';c.old.atomic_json(inp,model(schedule,free_float,mass,steps))
    start=time.perf_counter()
    result=subprocess.run([str(EP),'-w',str(thermal_weather()),'-d',str(directory),str(inp)],capture_output=True,text=True)
    (directory/'console.log').write_text(result.stdout+'\n'+result.stderr)
    err=(directory/'eplusout.err').read_text() if (directory/'eplusout.err').exists() else result.stdout+result.stderr
    if result.returncode or '** Severe **' in err or '**  Fatal  **' in err:raise RuntimeError(err[-7000:])
    df=pd.read_csv(directory/'eplusout.csv');assert len(df)==8760,(directory,len(df))
    def col(name):
        names=[x for x in df.columns if name.upper() in x.upper()]
        if not names:return np.zeros(8760)
        return df[names].sum(axis=1).to_numpy()
    h=col('Zone Ideal Loads Zone Sensible Heating Energy');co=col('Zone Ideal Loads Zone Sensible Cooling Energy')
    hr=col('Zone Ideal Loads Zone Sensible Heating Rate');cr=col('Zone Ideal Loads Zone Sensible Cooling Rate')
    native_names=[x for x in df.columns if 'SURFACE WINDOW TRANSMITTED SOLAR RADIATION ENERGY' in x.upper()]
    assert native_names,'Native solar diagnostic unavailable'
    native=col('Surface Window Transmitted Solar Radiation Energy')
    assert np.max(np.abs(native))<1e-6,('Native solar not zero',np.max(native))
    indoor=col('Zone Mean Air Temperature');temp=col('Zone Operative Temperature')
    data=dict(heating_kwh=float(h.sum()/3.6e6),cooling_kwh=float(co.sum()/3.6e6),peak_heating_w=float(hr.max()),peak_cooling_w=float(cr.max()),
        indoor_max_c=float(indoor.max()),operative_max_c=float(temp.max()),overheat_hours_26=int(np.sum(temp>26)),run_s=time.perf_counter()-start,
        schedule_solar_kwh=float(gains.sum()/1000),native_solar_kwh=float(native.sum()/3.6e6),native_shortwave_zero_driver=True,warnings=err.count('** Warning **'),free_float=free_float,mass=mass,steps_per_hour=steps)
    c.old.atomic_json(report,data);c.old.atomic_npz(directory/'hourly_outputs.npz',heating_j=h,cooling_j=co,heating_w=hr,cooling_w=cr,air_c=indoor,operative_c=temp)
    return data

def smoke():
    print(run_case(RESULTS/'thermal/smoke',np.zeros(8760)),flush=True)

def study():
    rows=[]
    manifest=json.loads((RESULTS/'frozen_hourly_manifest.json').read_text())
    settings=json.loads((RESULTS/'calibration_final.json').read_text())
    for item in manifest['receivers']:
        if not item['label'].startswith('wall'):continue
        path=RESULTS/'validation'/item['scene']/item['label']
        hourly=path/'hourly.npz'
        if not hourly.exists():continue
        z=np.load(hourly);meta=json.loads((path/'metadata.json').read_text())
        # Uniform equivalent glazing; exact and surrogate share all thermal assumptions.
        beam=np.maximum(z['dni'],0)*np.maximum(z['incidence'],0)*(z['el']>0);diff=np.maximum(z['dhi'],0)*(.5+.5*meta['normal'][2])
        for variant,col in [('exact',0),('lobe',1)]:
            gains=6*.5*(beam*(1-z['shade'][:,col])+diff)
            directory=RESULTS/'thermal'/item['scene']/item['label']/variant
            data=run_case(directory,gains)
            rows.append(dict(scene=item['scene'],label=item['label'],variant=variant,**data))
        print('THERMAL',item['scene'],item['label'],flush=True)
        c.old.atomic_csv(RESULTS/'thermal_results.csv',rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','study']);args=parser.parse_args()
    smoke() if args.mode=='smoke' else study()
