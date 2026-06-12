import logging
import pandas as pd
import random

CSV_PATH = "datasets/obelis_data/df_lv.csv"
FEATHER_PATH = "datasets/obelis_data/df_lv.feather"

logger = logging.getLogger(__name__)


class Data_Provider():
    """Abstract class to allow implementation of different data provider strategies, providing the datasets for non observable vehicle spawns and despawns."""
    def get_non_observable_vehicle_data(self, weekday=None):
        raise NotImplementedError


class Obelis_Data_Provider(Data_Provider):
    """
    Providing the OBELIS dataset (from Germany) to supply the data for non observable vehicle spawns and despawns.
    Sessions are pooled per weekday (Monday=0 … Sunday=6) at init time and sampled on each call.
    """

    def __init__(self, target_session_count, seed=None):
        self.lp_ids = ['6804_shuffled', '6973_shuffled', '16564_shuffled', '6326_shuffled']
        self.target_session_count = target_session_count
        self.rng = random.Random(seed)

        df = pd.read_feather(FEATHER_PATH)
        df = df[df["lp_id"].isin(self.lp_ids)]

        self.pools_by_weekday = {}
        for wd in range(7):
            df_wd = df[df["beginn"].dt.dayofweek == wd]
            self.pools_by_weekday[wd] = self.extract_non_observable_vehicle_data(df_wd)
            logger.info("Weekday %d pool: %s sessions", wd, len(self.pools_by_weekday[wd]))

    def __match_cs(self, lp_id):
        cs_matching_dict = {
            self.lp_ids[0]: "cs_1",
            self.lp_ids[1]: "cs_2",
            self.lp_ids[2]: "cs_3",
            self.lp_ids[3]: "cs_4"
        }
        return cs_matching_dict[lp_id]

    def __convert_to_depart_time(self, charge_begin_datetime):
        hours = charge_begin_datetime.hour
        minutes = charge_begin_datetime.minute
        seconds = charge_begin_datetime.second
        return (hours * 3600) + (minutes * 60) + seconds

    def __convert_data_from_csv_row(self, lp_id, charge_begin_datetime, charge_duration):
        cs_id = self.__match_cs(lp_id)
        charge_begin_seconds = self.__convert_to_depart_time(charge_begin_datetime)
        result_dict = {
            "cs_id": cs_id,
            "charge_begin_seconds": charge_begin_seconds,
            "charge_duration": charge_duration,
        }
        return result_dict

    def extract_non_observable_vehicle_data(self, dataframe):
        vehicle_data = []
        for row in dataframe.itertuples(index=False):
            result_dict = self.__convert_data_from_csv_row(row.lp_id, row.beginn, row.dauer_sekunden)
            vehicle_data.append(result_dict)
        return vehicle_data

    def get_non_observable_vehicle_data(self, weekday):
        if weekday is None:
            weekday = self.rng.randint(0, 6)
            logger.debug("No episode date available; using random weekday %d for NOEV sampling.", weekday)
        pool = self.pools_by_weekday[weekday]
        if not pool:
            logger.warning("Weekday %d has no NOEV sessions in pool; returning empty list.", weekday)
            return []
        if len(pool) >= self.target_session_count:
            return self.rng.sample(pool, self.target_session_count)
        logger.warning(
            "Weekday %d pool has %s sessions, fewer than target %s; using all available.",
            weekday, len(pool), self.target_session_count,
        )
        return list(pool)


class Random_Data_Provider(Data_Provider):
    """
    Providing random data to supply the data for non observable vehicle spawns and despawns.
    Params:
        n_noevs (int): The amount of non observable vehicles which should be simulated
        n_cs (int): The amount of charging stations in the system
        max_simulation_time (int): The expected max duration of the simulation in seconds
    """
    def __init__(self, n_noevs, n_cs, max_simulation_time, seed=None):
        self.rng = random.Random(seed)
        self.n_noevs = n_noevs
        self.n_cs = n_cs
        self.max_simulation_time = max_simulation_time
        super().__init__()

    def get_non_observable_vehicle_data(self, weekday=None):
        vehicle_data = []
        for i in range(self.n_noevs):
            cs_id = f"cs_{self.rng.randint(1, self.n_cs)}"  # station IDs are 1-based (cs_1..cs_n)
            begin = self.rng.randint(0, self.max_simulation_time)  # spawn time in seconds after simulation start
            duration = self.rng.randint(10, 200)  # charge duration in seconds
            entry_dict = {
                "cs_id": cs_id,
                "charge_begin_seconds": begin,
                "charge_duration": duration,
            }
            vehicle_data.append(entry_dict)
        return vehicle_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_provider = Obelis_Data_Provider(target_session_count=5, seed=42)
    vehicle_data = data_provider.get_non_observable_vehicle_data(weekday=0)  # Monday
    print(vehicle_data)
