import pandas as pd

CSV_PATH="obelis_data/df_lv.csv"

class Obelis_Data_Processor():


    def __init__(self) -> None:
        self.lp_ids = ['6804_shuffled', '6973_shuffled', '16564_shuffled', '6326_shuffled']
        self.df = pd.read_csv(CSV_PATH, delimiter=";", parse_dates=["beginn", "ende"])

    
    def __prepare_dataframe(self, dataframe, filter_date):
        # include only the lp_ids specified in self.lp_ids
        df_filtered = dataframe[dataframe["lp_id"].isin(self.lp_ids)]
        # include only the day in question
        df_filtered = df_filtered[df_filtered["beginn"].dt.date == pd.to_datetime(filter_date).date()]
        return df_filtered

    def __match_cs(self, lp_id):
        cs_matching_dict = {
            self.lp_ids[0]: "cs_0",
            self.lp_ids[1]: "cs_1",
            self.lp_ids[2]: "cs_2",
            self.lp_ids[3]: "cs_3"
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
    
    def extract_non_member_vehicle_data(self, dataframe):
        vehicle_data = []
        for row in dataframe.itertuples(index=False):
            result_dict = self.__convert_data_from_csv_row(row.lp_id, row.beginn, row.dauer_sekunden)
            vehicle_data.append(result_dict)
        return vehicle_data

    def get_non_member_vehicle_data(self):
        filter_date = '2023-03-05'
        df = self.df
        df_filtered = self.__prepare_dataframe(df, filter_date)
        vehicle_data = self.extract_non_member_vehicle_data(df_filtered)
        return vehicle_data


if __name__ == "__main__":
    vehicle_data = Obelis_Data_Processor().get_non_member_vehicle_data()
    print(vehicle_data)