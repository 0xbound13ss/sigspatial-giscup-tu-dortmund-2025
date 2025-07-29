from config import settings
from geobleu_seq_eval import calc_geobleu_humob25, calc_dtw_humob25


def get_algo() -> callable:
    if settings.ALGO_USED == "geo_bleu":
        return calc_geobleu_humob25
    elif settings.ALGO_USED == "dtw":
        return calc_dtw_humob25


def get_xy_list_from_df(df, uid):
    user_df = df[df["uid"] == uid]
    return list(zip(user_df["x"].astype(int), user_df["y"].astype(int)))


def get_dxy_list_from_df(df, uid):
    user_df = df[df["uid"] == uid]
    return list(zip(user_df["dx"].astype(int), user_df["dy"].astype(int)))


def get_xy_list_from_df_simple(df):
    return list(zip(df["x"].astype(int), df["y"].astype(int)))
