import pandas as pd
import xgboost as xgb
def map_for_dict_Gender(Gender):
    dict_Gender = {'Male': 0, 'Female': 1}
    res = dict_Gender.get(Gender)
    return res


def map_for_dict_MariStat(MariStat):
    dict_MariStat = {'Other': 0, 'Alone': 1}
    res = dict_MariStat.get(MariStat)
    return res


def f_VehUsage_Professional(VehUsage):
    if VehUsage == 'Professional':
        VehUsage_Professional = 1
    else:
        VehUsage_Professional = 0
    return VehUsage_Professional

def sqrt_driveAge(DrivAge):
    return DrivAge*DrivAge


def f_VehUsage_Private_trip_to_office(VehUsage):
    if VehUsage == 'Private+trip to office':
        VehUsage_Private_trip_to_office = 1
    else:
        VehUsage_Private_trip_to_office = 0
    return VehUsage_Private_trip_to_office


def f_VehUsage_Private(VehUsage):
    if VehUsage == 'Private':
        VehUsage_Private = 1
    else:
        VehUsage_Private = 0
    return VehUsage_Private


def f_VehUsage_Professional_run(VehUsage):
    if VehUsage == 'Professional run':
        VehUsage_Professional_run = 1
    else:
        VehUsage_Professional_run = 0
    return VehUsage_Professional_run


def return_pd_Frame():
    columns = [
        'LicAge',
        'Gender',
        'MariStat',
        'DrivAge',
        'HasKmLimit',
        'BonusMalus',
        'OutUseNb',
        'RiskArea',
        'VehUsg_Private',
        'VehUsg_Private+trip to office',
        'VehUsg_Professional',
        'VehUsg_Professional run',
        'SocioCateg_CSP1',
        'SocioCateg_CSP2',
        'SocioCateg_CSP3',
        'SocioCateg_CSP4',
        'SocioCateg_CSP5',
        'SocioCateg_CSP6',
        'SocioCateg_CSP7',
        'DrivAgeSq'
    ]

    df1=pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],index=columns)
    df2=df1.T

    return df2

def process_input(json_input):
    LicAge = json_input["LicAge"]
    Gender = map_for_dict_Gender(json_input["Gender"])
    MariStat = map_for_dict_MariStat(json_input["MariStat"])
    DrivAge = json_input["DrivAge"]
    DrivAgeSqrt = sqrt_driveAge(json_input["DrivAge"])
    HasKmLimit = json_input["HasKmLimit"]
    BonusMalus = json_input["BonusMalus"]
    OutUseNb = json_input["OutUseNb"]
    RiskArea = json_input["RiskArea"]
    VehUsg_Private = f_VehUsage_Private(json_input["VehUsage"])
    VehUsg_Private_trip_to_office = f_VehUsage_Private_trip_to_office(json_input["VehUsage"])
    VehUsg_Professional = f_VehUsage_Professional(json_input["VehUsage"])
    VehUsg_Professional_run = f_VehUsage_Professional_run(json_input["VehUsage"])
    CSP='SocioCateg_'+json_input["SocioCateg"][:4].upper()



    df2=return_pd_Frame()
    df2['LicAge'] = LicAge
    df2['Gender'] = Gender
    df2['MariStat'] = MariStat
    df2['DrivAge'] = DrivAge
    df2['HasKmLimit'] = HasKmLimit
    df2['BonusMalus'] = BonusMalus
    df2['OutUseNb'] = OutUseNb
    df2['RiskArea'] = RiskArea
    df2['VehUsg_Private'] = VehUsg_Private
    df2['VehUsg_Private+trip to office'] = VehUsg_Private_trip_to_office
    df2['VehUsg_Professional'] = VehUsg_Professional
    df2['VehUsg_Professional run'] = VehUsg_Professional_run
    df2['DrivAgeSq'] = DrivAgeSqrt

    df2[CSP] = 1
    df3 = xgb.DMatrix(df2)

    return df3