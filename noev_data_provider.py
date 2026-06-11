import pandas as pd
import random

CSV_PATH="datasets/obelis_data/df_lv.csv"
FEATHER_PATH="datasets/obelis_data/df_lv.feather"

class Data_Provider():
    """Abstract class to allow implementation of different data provider strategies, providing the datasets for non observable vehicle spawns and despawns."""
    def get_non_observable_vehicle_data(self):
        raise NotImplementedError

class Obelis_Data_Provider(Data_Provider):
    """
    Providing the OBELIS dataset (from Germany) to supply the data for non observable vehicle spawns and despawns.
    """

    def __init__(self) -> None:
        self.lp_ids = ['6804_shuffled', '6973_shuffled', '16564_shuffled', '6326_shuffled']
        # self.df = pd.read_csv(CSV_PATH, delimiter=";", parse_dates=["beginn", "ende"])
        self.df = pd.read_feather(FEATHER_PATH)

    
    def __prepare_dataframe(self, dataframe, filter_date):
        # include only the lp_ids specified in self.lp_ids
        df_filtered = dataframe[dataframe["lp_id"].isin(self.lp_ids)]
        # include only the day in question
        df_filtered = df_filtered[df_filtered["beginn"].dt.date == pd.to_datetime(filter_date).date()]
        return df_filtered

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
            "charge_duration": charge_duration
        }
        return result_dict
    
    def extract_non_observable_vehicle_data(self, dataframe):
        vehicle_data = []
        for row in dataframe.itertuples(index=False):
            result_dict = self.__convert_data_from_csv_row(row.lp_id, row.beginn, row.dauer_sekunden)
            vehicle_data.append(result_dict)
        return vehicle_data

    def get_non_observable_vehicle_data(self):
        filter_date = '2023-03-05'
        df = self.df
        df_filtered = self.__prepare_dataframe(df, filter_date)
        vehicle_data = self.extract_non_observable_vehicle_data(df_filtered)
        return vehicle_data
    
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

    def get_non_observable_vehicle_data(self):
        vehicle_data = []
        for i in range(self.n_noevs):
            cs_id = f"cs_{self.rng.randint(1, self.n_cs)}"  # station IDs are 1-based (cs_1..cs_n)
            begin = self.rng .randint(0, self.max_simulation_time)# spawn time in seconds after simulation start
            duration = self.rng .randint(10, 200) # charge duration in seconds
            entry_dict = {
                "cs_id": cs_id,
                "charge_begin_seconds": begin,
                "charge_duration": duration
            }
            vehicle_data.append (entry_dict)
        return vehicle_data


if __name__ == "__main__":
    data_provider = Obelis_Data_Provider()
    # data_provider = Random_Data_Provider(5, 4, 3000)
    vehicle_data = data_provider.get_non_observable_vehicle_data()
    print(vehicle_data)