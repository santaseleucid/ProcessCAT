from ProcessCAT import *
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()
root.attributes('-topmost', True)

input("Enter any key to choose folder containing raw data\n")
data_directory = filedialog.askdirectory()
print("DATA DIRECTORY: ", data_directory, "\n")

input("Enter any key to choose folder to export files to\n")
export_directory = filedialog.askdirectory()
print("EXPORT DIRECTORY: ", export_directory, "\n")


CAT = ProcessCAT(data_directory, export_directory)
CAT.set_summary()
CAT.get_summary()

rms_csv = input("Export 1m RMS CSV? [y/n]\n")
rms_png = input("Export 1m RMS PNG? [y/n]\n")
oct3_csv = input("Export 3rd OCT CSV? [y/n]\n")
oct3_png = input("Export 3rd OCT PNG? [y/n]\n")
oct24_csv = input("Export 24th OCT CSV? [y/n]\n")
oct24_png = input("Export 24th OCT PNG? [y/n]\n")
octavefracs = []
if(oct3_csv == 'y' or oct3_png == 'y'):
    octavefracs.append(3)
if(oct24_csv == 'y' or oct24_png == 'y'):
    octavefracs.append(24)
CAT.set_exports(
    rms_csv=rms_csv == 'y',
    rms_png=rms_png == 'y',
    oct3_csv=oct3_csv == 'y',
    oct3_png=oct3_png == 'y',
    oct24_csv=oct24_csv == 'y',
    oct24_png=oct24_png == 'y',
    octavefracs=octavefracs
)

CAT.process_driver()
