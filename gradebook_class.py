# Provides module gradebook
from pathlib import Path
import pandas as pd
import re
import math
import numpy
import matplotlib.pyplot as plt
# import plotext as plt
from tabulate import tabulate
# Some helper functions first


def convname(s):
    """ Helper function to convert two most common types of name formats"""
    if ", " in s:
        t = s.split(", ")
        second_name = t[0]
        first_name = t[1:]
        return " ".join([first_name, second_name])
    elif " " in s:
        t = s.split()
        second_name = t[-1]
        first_name = " ".join(t[:-1])
        return ", ".join([second_name, first_name])
    else:
        return -1



def round_to_multiple(x, multiple):
    try:
        return multiple*round(x/multiple)
    except Exception:
        return numpy.nan


def standard_marking_scale(points, points_4, points_6):
    """ Returns mark given the points needed for a mark of 4 and 6, respectively.

    Note: The function first rounds to a multiple of 0.5
    before using the double-linear scale to obtain a mark
    in the range of 1 to 6; the output is rounded to
    multiples of 0.1.
    (all rounding are such that the "middle" is rounded up)
    """
    # round to half points
    points = round_to_multiple(points, 0.5)
    if points <= points_4:
        return round(round_to_multiple(1 + points/points_4*3, 0.1), ndigits=1)
    elif points > points_4:
        if points >= points_6:
            return 6.0
        else:
            return round(round_to_multiple(4 + (points-points_4)/(points_6 - points_4) * 2, 0.1), ndigits=1)
    else:
        print("Something is wrong")
        return -1


def triple_marking_scale(points, points_3, points_4, points_6):
    """ Returns mark given the points needed for a mark of 3, 4 and 6, respectively.

    Note: The function first rounds to a multiple of 0.5
    before using the triple-linear scale to obtain a mark
    in the range of 1 to 6; the output is rounded to
    multiples of 0.1.
    (all rounding are such that the "middle" is rounded up)
    """

    # round to half points
    points = round_to_multiple(points, 0.5)
    if points <= points_3:
        return round(round_to_multiple(1 + points/points_3*2, 0.1), ndigits=1)
    elif points <= points_4:
        return round(round_to_multiple(3 + (points-points_3)/(points_4-points_3)*1, 0.1), ndigits=1)
    elif points > points_4:
        if points >= points_6:
            return 6.0
        else:
            return round(round_to_multiple(4 + (points-points_4)/(points_6 - points_4) * 2, 0.1), ndigits=1)
    else:
        print("Something is wrong")
        return -1


class Notenbuch:
    def __init__(self, title):
        self.title = title
        self.grades = []
        self.weights = []
        self.roster = -1

    def add(self, ln):
        if isinstance(ln, Leistungsnachweis):
            self.grades.append(ln)
        else:
            print(
                "Fehler: Du versuchtst etwas anzuhängen, was kein Leistungsnachweis ist!")

    def load_roster(self, path):
        df = pd.read_csv(path,
                         usecols=["Nachname", "Vorname", "EMail"],
                         )
        df["Name"] = df["Nachname"]+", "+df["Vorname"]
        self.roster = df

    def merge_grades(self, method="stdout"):
        """ Merge grades in book

        """

        from thefuzz import process, fuzz
        frames = [x.points[["Name", "Note"]].rename(
            {"Note": x.get_shortname()}, axis=1) for x in self.grades]

        for frame in frames:
            frame["Name"] = frame["Name"].apply(lambda x: process.extractOne(
                x, self.roster["Name"], scorer=fuzz.partial_ratio)[0])

        out = None
        from functools import reduce
        try:
            out = reduce(lambda left, right: pd.merge(
                left, right, on="Name", how="outer"), frames)
            out = pd.merge(self.roster, out, on="Name", how="left")
        except:
            print("No marks set")

        out = out.reset_index().drop("index", axis=1)
        out.index += 1
        self.overview = out

    def get_overview(self, mode="simple", sortby="Name"):
        """
        Function to get overview in condensed form.
        Different modes are available.
        Default view "simple" only shows Name in Form, "Nachname, Vorname".
        "Names" shows both first and second name.
        "Full" also shows EMail and both name variants.
        """
        if mode == "simple":
            if sortby in ["Nachname", "Vorname"]:
                print("In mode \"simple\" you can only sort by full names")
                return -1
            return self.overview.drop(columns=["Nachname", "Vorname", "EMail"]).sort_values(by=sortby)
        if mode == "names":
            if sortby == "Name":
                sortby = "Nachname"
            return self.overview.drop(
                columns=["Name", "EMail"]).sort_values(by=sortby)
        if mode == "full":
            return self.overview.sort_values(by=sortby)

    def set_weights(self, weights):
        self.merge_grades()
        self.weights = pd.DataFrame(
            weights, index=self.overview.drop(columns=["Name", "Vorname", "Nachname", "EMail"]).keys(), columns=["weights"])

    def weighted_average(self, vals):
        eff_sum_of_weights = sum(w for (x, w) in zip(
            vals, self.weights["weights"]) if not pd.isna(x))
        eff_sum = sum(x*w for (x, w) in zip(vals,
                                            self.weights["weights"]) if not pd.isna(x))
        return eff_sum/eff_sum_of_weights

    def auto_calc_grade(self):
        if "Schnitt" in self.overview.keys():
            print("Grade has been already calculated")
            while True:
                tf = input("Recalculate grades? (y/n)")
                if tf == "n":
                    print("Abort")
                    return None
                elif tf not in ["y", "n"]:
                    pass
                elif tf == "y":
                    self.merge_grades()
                    break
        if len(self.weights) == 0:
            print("No weights set. Calculating simple average")
            self.overview["Schnitt"] = round(self.overview.drop(
                columns=["Nachname", "Vorname", "Name", "EMail"]).mean(axis=1), 2)

        else:
            # apply a weighing function to a combination of columns
            self.overview["Schnitt"] = round(self.overview.apply(lambda x: self.weighted_average(
                [x[col] for col in self.overview.drop(columns=["Nachname", "Vorname", "Name", "EMail"]).keys()]), axis=1), 2)

    def calc_final_grade(self):
        if not "Schnitt" in self.overview.keys():
            print(
                "No grades calculated yet. Calculate them manually or use auto_calc_grade()")
            return -1
        else:
            self.overview["Note"] = self.overview["Schnitt"].apply(
                    lambda x: round_to_multiple(x, 0.5))


class Leistungsnachweis:
    def __init__(self, path, title):
        self.title = title
        self.path = path
        self.notes = {}
        self.scale_params = {}

    def get_shortname(self):
        p = re.compile('[a-zA-Züäüö]*[1-9]')
        m = p.match(self.title.replace(" ", ""))
        str_to_return = self.title[0]+self.title.replace(" ", "")[m.end()-1]
        if self.title.find("NT") != -1:
            str_to_return += "_NT"
        return str_to_return

    def get_marks(self):
        try:
            return self.points["Note"].rename(self.title)
        except:
            print("Set marks first")

    def clean_data(self):
        """
        Cleans data of students that did not partake in exam.
        It removes those rows from "points"-DataFrame
        """
        # drop all columns not related to points of questions
        t = self.points.drop(
            ["Name", "Vorname", "Nachname", "Summe"], axis=1, errors="ignore")
        # get number of nans in row
        num_of_nans_in_row = t.isna().sum(axis=1)
        # index df with True only if no point data was available
        log_df = num_of_nans_in_row == t.shape[1]
        # cumbersome way of removing those rows that did not have point data, but only nans there
        self.points = pd.concat([self.points[log_df], self.points]
                                ).drop_duplicates(keep=False).reset_index().drop("index", axis=1)
        # reindexing so, that index is 1...num of students
        self.points.index += 1

    def print_info(self):
        print(tabulate(self.info, tablefmt="orgtbl",
              headers=[self.info.index.name]+list(self.info.keys())))

    def sum_points(self):
        t = self.points.drop(
            ["Name", "Vorname", "Nachname", "EMail"], axis=1, errors="ignore")
        if "Summe" in t.columns:
            print("Sum already calculated")
            return -1
        else:
            self.points["Summe"] = t.sum(axis=1)

    def set_marks(self, option="standard", params=[]):
        if option == "standard":
            try:
                p4, p6 = params
                self.scale_params["Skalentyp"] = option
                self.scale_params["Punkte für Note 4"] = p4
                self.scale_params["Punkte für Note 6"] = p6
                self.points["Note"] = self.points["Summe"].apply(
                    lambda x: standard_marking_scale(x, p4, p6))
            except:
                print("Falsche Parameter für diese Notenskala:" + option)

        elif option == "triple":
            try:
                p3, p4, p6 = params
                self.scale_params["Skalentyp"] = option
                self.scale_params["Punkte für Note 3"] = p3
                self.scale_params["Punkte für Note 4"] = p4
                self.scale_params["Punkte für Note 6"] = p6
                self.points["Note"] = self.points["Summe"].apply(
                    lambda x: triple_marking_scale(x, p3, p4, p6))
            except:
                print("Falsche Parameter für diese Notenskala:" + option)
        else:
            print("keine valide Option. Überprüfe die Schreibweise")

    def print_gradescale(self, xmin=-1):
        grades_list = []
        table = []
        if xmin == -1:
            x_l = math.floor(self.points["Summe"].min())
        else:
            x_l = xmin
        x_u = self.scale_params["Punkte für Note 6"]
        pointslist = numpy.arange(x_l, x_u+.5, .5)
        if self.scale_params["Skalentyp"] == "standard":
            for x in pointslist:
                table.append((str(x),
                              str(standard_marking_scale(x,
                                                         self.scale_params["Punkte für Note 4"],
                                                         self.scale_params["Punkte für Note 6"]))))

        elif self.scale_params["Skalentyp"] == "triple":
            for x in pointslist:
                table.append((str(x),
                              str(triple_marking_scale(x,
                                                       self.scale_params["Punkte für Note 3"],
                                                       self.scale_params["Punkte für Note 4"],
                                                       self.scale_params["Punkte für Note 6"]))))

        print(tabulate(table, headers=[
              "Punkte", "Note"], tablefmt="orgtbl"))

    def plot_hist(self, title="default"):
        plt.figure(num=title, clear=True)
        seq = numpy.arange(.75, 6.75, .5)
        self.get_marks().hist(bins=seq, rwidth=.7)
        plt.show(block=False)

    def print_stats(self, option="grades"):
        if option in ["points", "all"]:
            table = [["Punkte", self.info["max_Punkte"].agg("sum")],
                     ["max erreichte Punkte", round(
                         self.points["Summe"].agg("max"), 1)],
                     ["min erreichte Punkte", round(
                         self.points["Summe"].agg("min"), 1)],
                     ["mittlere Punktzahl", round(
                         self.points["Summe"].agg("mean"), 1)],
                     ["Median der Punkte", round(
                         self.points["Summe"].agg("median"), 1)],
                     ["Standardabw. Punkte", round(self.points["Summe"].agg("std"), 1)]]
            print(tabulate(table, headers=["", "Punkte"], tablefmt="orgtbl"))

        if option in ["grades", "all"]:
            try:
                table = [["Max", round(self.points["Note"].agg("max"), 1)],
                         ["Min", round(self.points["Note"].agg("min"), 1)],
                         ["Durchschnitt", round(
                             self.points["Note"].agg("mean"), 1)],
                         ["Median", round(
                             self.points["Note"].agg("median"), 1)],
                         ["Standardabw.", round(self.points["Note"].agg("std"), 1)]]
                print(tabulate(table, headers=["", "Note"], tablefmt="orgtbl"))
            except:
                print("Es ist ein Fehler aufgetreten. Zuerst Noten setzen.")
    def print_grades(self, sort = "Index"):
        t = self.points.drop("Name", axis=1)
        if sort == "Index":
            print(tabulate(t.sort_index(), tablefmt = "orgtbl", headers= t.keys()))
        else:
            print(tabulate(t.sort_values(by=sort), tablefmt = "orgtbl", headers= t.keys()))


class EinfacheNote(Leistungsnachweis):
    """
    EinfacheNote(path, title)

    Note setzen.

    path: Import aus .csv-Datei in "path"
    title: string zur Identifizierung

    Die csv-Datei sollte die Spalten "Nachname", "Vorname" und "Note" enthalten

    """

    def __init__(self, path, title):
        super().__init__(path, title)
        self._data = self._import_raw_data()
        self.points = self._extract_to_points_df()
        self.clean_data()

    def _import_raw_data(self):
        "Daten aus csv für Prüfung lesen"

        try:
            df = pd.read_csv(
                self.path,
                dtype={"Note": numpy.float64},
            )
        except:
            print("Error.")
            print("Check wether you provided a correct path to the .csv-file")
            print("Check wether you provided a .csv-file containing columns \"Nachname\", \"Vorname\" and \"Note\" ")
        return df

    def _extract_to_points_df(self):
        df_points = self._data.copy()
        # convert name to format "second_name, first_name" if necessary

        if "Vorname" in df_points.columns:
            if "Name" in df_points.columns:
                df_points = df_points.rename(columns={"Name": "Nachname"})
            df_points["Name"] = df_points["Nachname"] + \
                ", " + df_points["Vorname"]
        # index = 1,2,3,...
        df_points.index += 1
        df_points = df_points[["Nachname", "Vorname", "Name", "Note"]]
        return df_points


class Pruefung(Leistungsnachweis):
    def __init__(self, path, title):
        super().__init__(path, title)
        self._data = self._import_raw_data()
        self.points = self._extract_to_points_df()
        self.clean_data()
        # # sum points after cleaning data of students that did not hand in test
        self.sum_points()
        self.info = self._extract_to_info_df()
        self.scale_params = {}

    def _import_raw_data(self):
        "Daten aus csv für Prüfung lesen"

        df = pd.read_csv(
            self.path,
            usecols=lambda x: "Unnamed" not in x
        )
        return df

    def _extract_to_points_df(self):

        df_points = self._data.copy()
        if "Vorname" in df_points.columns:
            if "Name" in df_points.columns:
                df_points.rename(columns={"Name": "Nachname"}, inplace=True)
            df_points["Name"] = df_points["Nachname"] + \
                ", " + df_points["Vorname"]
            # df_points.drop("Vorname", axis=1, inplace=True)
            # reorder columns
            t = list(df_points.columns)
            t.remove("Vorname")
            t.remove("Nachname")
            t.remove("Name")
            new_order = ["Name", "Nachname", "Vorname"]+t
            df_points = df_points[new_order]

        else:
            if not "," in df_points["Name"]:
                df_points["Name"] = df_points["Name"].apply(convname)
            # df_points["Vorname"] = df_points["Name"].apply(
            #    lambda x: x.split(",")[-1].replace(" ", ""))
        # drop second line (=max points) and possibly last line using merge
        # i.e.: drop if "Name" is NaN
        df_points = df_points.merge(
            df_points[["Name"]].dropna(how="any"), "inner")
        df_points.reset_index().drop(columns="index")
        df_points.index += 1
        return df_points

    def _extract_to_info_df(self):
        # extract meta infos from complete table
        mp_series = self._data.iloc[0, 2:]
        err_pkt = [round(self.points[x].mean() / mp_series.loc[x]*100, 0)
                   for x in mp_series.index]
        df_info = pd.DataFrame(
            {"Max_Punkte": mp_series, "Erreichte Punkte": err_pkt})
        return df_info


class Classtime(Leistungsnachweis):
    def __init__(self, path, title):
        super().__init__(path, title)
        self.path = path
        self._data = self._import_raw_data()
        self.points = self._extract_to_points_df()
        self.clean_data()
        # sum points after cleaning data of students that did not hand in test
        self.sum_points()
        self.info = self._extract_to_info_df()
        self.scale_params = {}

    def clean_data(self):
        """
        Cleans data of students that did not partake in exam.
        It removes those rows from "points"-DataFrame
        """
        # drop all columns not related to points of questions
        t = self.points.drop(
            ["Name", "Vorname", "Nachname", "Note", "Summe"], axis=1, errors="ignore")
        # get number of nans in row
        num_of_nans_in_row = t.isna().sum(axis=1)
        # index df with True only if no point data was available
        log_df = num_of_nans_in_row == t.shape[1]
        # cumbersome way of removing those rows that did not have point data, but only nans there
        self.points = pd.concat([self.points[log_df], self.points]
                                ).drop_duplicates(keep=False).reset_index().drop("index", axis=1)
        # reindexing so, that index is 1...num of students
        self.points.index += 1

    def _import_raw_data(self):

        df = pd.read_excel(
            self.path,
            sheet_name=self.path.split("/")[-1].split(".")[0],
            usecols=lambda x: "Unnamed" not in x,
        )
        return df

    def _extract_to_points_df(self):

        df = self._data.copy()
        # rename it to just "Q1", "Q2" and so on
        df.rename(inplace=True, columns=lambda x: x[x.rfind("Q"):])
        # extract only infos about points achieved in each exercise
        df_points = df.drop(index=range(4), axis=0)
        # rename first column to "Name"
        df_points.rename(columns={df_points.columns[0]: "Name"}, inplace=True)
        # convert name to format "second_name, first_name"
        df_points["Name"] = df_points["Name"].apply(convname)
        df_points = df_points.reset_index()
        df_points.index += 1
        df_points = df_points.drop(columns=["index"])
        # df_points.set_index("Name", inplace=True)
        # df_points.sort_index(inplace=True)
        return df_points

    def _extract_to_info_df(self):
        # extract meta infos from complete table
        df_info = self._data.iloc[[0, 3], 1:].transpose()
        # reg exp to find date string in first row
        # doesn't seem to work on new machine (fedora), maybe different handling of importer
        # longlabel is hidden in comments in the spreadsheet
        # -> in Index there is not hint of the comments on the fedora machine, whereas
        # on the old ubuntu the index is a combination of the comment and the cell!
        df_info.rename(columns={0: "Fragentyp", 3: "max_Punkte"}, inplace=True)
        shortlabels = [x[x.rfind("Q"):] for x in df_info.index]
        p = re.compile(
            '[1-9][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
        try:
            longlabels = [x[p.match(x).end():x.rfind("Q")] for x in df_info.index]
        except:
            print("Info: Titel der Aufgaben konnte nicht gelesen werden. Setze Kurztitel ebenfalls auf Titel")
            longlabels = shortlabels
        df_info["Frage"] = shortlabels
        df_info.set_index("Frage", inplace=True)
        df_info["Titel"] = longlabels
        df_info = df_info[["Titel", "Fragentyp", "max_Punkte"]]
        df_info["Erreichte Punkte"] = [round(self.points[x].mean()/df_info.loc[x]["max_Punkte"]*100, 0)
                                       for x in df_info.index]
        return df_info


class MoodleTest(Leistungsnachweis):
    def __init__(self, path, title):
        super().__init__(path, title)
        self.path = path
        self._data = self._import_raw_data()
        self.points = self._extract_to_points_df()
        # Moodle exports "-" for answers that haven't been sent
        self.points.replace(to_replace="-", value="0", inplace=True)
        self.clean_data()
        # sum points after cleaning data of students that did not hand in test
        self.points["Summe"] = self.points.sum(
            axis=1, numeric_only=True).values
        self.info = self._extract_to_info_df()
        self.scale_params = {}

    def _import_raw_data(self):

        df = pd.read_csv(
            self.path,
            engine="python",  # to use skipfooter, this is necessary
            skipfooter=1,
            # necessary; otherwise columns with "-" become of dtype string
            na_values=["-"]
        )
        return df

    def _extract_to_points_df(self):

        df_points = self._data.copy()
        # rename it to just "F1", "F2" and so on

        def temp_name_replace(s):
            pos = s.rfind("/")
            if pos != -1:
                return s[:pos].replace(" ", "")
            else:
                return s
        df_points.rename(inplace=True, columns=temp_name_replace)
        # extract only infos about points achieved in each exercise
        df_points.drop(columns=["Status", "Begonnen",
                                "Beendet", "Dauer", "Bewertung"], inplace=True)
        # rename E-Mail-Adresse
        df_points.rename(columns={"E-Mail-Adresse": "EMail"}, inplace=True)
        # add a column with Name, Vorname (necessary?)
        df_points["Name"] = df_points["Nachname"] + ", " + df_points["Vorname"]
        # reset_index and start with 1
        df_points = df_points.reset_index().drop(columns=["index"])
        df_points.index += 1
        return df_points

    def _extract_to_info_df(self):
        # extract meta infos from complete table
        p = re.compile(
            'F\s[0-9]+\s')
        labels = [s.split("/")[0].replace(" ", "")
                  for s in self._data.columns if not p.match(s) == None]
        points = [float(s.split("/")[1].replace(" ", ""))
                  for s in self._data.columns if not p.match(s) == None]
        df_info = pd.DataFrame({"max_Punkte": points}, index=labels)
        df_info["Erreichte Punkte"] = [round(self.points[x].mean()/df_info.loc[x]["max_Punkte"]*100, 0)
                                       for x in df_info.index]
        return df_info

    def export_for_moodle(self):
        return self.points.set_index("EMail").loc[:, "Note"].to_csv(self.title+"_to_moodle.csv")
