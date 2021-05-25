import os
import sys
import numpy as np
from spike_removal import *
from scipy import signal
import matplotlib.pyplot as plt
import math
from octave_standards import *


class ProcessCAT:

    def __init__(self, root_path, export_path):
        self.root_path = root_path
        self.export_path = export_path
        self.model = 1  # by default. get_sum_model will switch this if needed
        self.sum_obj_arr = []
        self.file_list = []
        self.model2_description_tag = False
        return

    def process_driver(self):
        # iterate through files in directory
        for sum_obj in self.sum_obj_arr:
            print("SUM FILE: " + sum_obj['filename'])
            for rail, sp in zip(sum_obj['rail'], sum_obj['sensor_position']):
                print("Processing ... " + rail, sp)
                if(len(sum_obj['rail']) == 2):
                    data_fname = sum_obj['filename'].replace(
                        '.sum', '_'+rail+'.txt')
                else:
                    data_fname = sum_obj['filename'].replace('.sum', '.txt')

                # Open File
                try:
                    raw_data = np.genfromtxt(
                        os.path.join(self.root_path, data_fname)
                    )

                except FileNotFoundError:
                    print(data_fname, ' not found, ensure path is correct')
                    exit(-1)
                if(raw_data.shape[0] == 0):
                    print(data_fname, " FILE EMPTY")
                    continue

                if(np.max(raw_data[:, 0]) == np.min(raw_data[:, 0])):
                    # need to generate new points because all zeros
                    x = input("generating distance at " + sum_obj['sampling_distance'] +
                              " mm because data file contains all zeros for distances.\n Enter any key to continue\n")
                    raw_data[:, 0] = np.arange(0, int(
                        sum_obj['sampling_distance'])*raw_data.shape[0], int(sum_obj['sampling_distance']))
                    raw_data[:, 0] = raw_data[:, 0]/1e6  # convert to km

                # Spike Removal
                disp = spike_removal(
                    raw_data[:, 1],
                    raw_data[:, 0]
                )

                disp[:1000] = 0
                disp[-1000:] = 0
                # + " - " + str(sum_obj['description'])
                plot_title = sum_obj['start_date'] + \
                    " - " + rail + " - " + str(sp)
                self.block_rms_driver(data_fname, raw_data[:, 0], disp, plot_title, int(
                    sum_obj['sampling_distance']))
                self.gen_spectrum_driver(
                    data_fname, raw_data[:, 0], disp, plot_title)
                print()
        return

    ###################################################
    # .SUM FILE PROCESSING
    ###################################################

    def set_exports(self, octavefracs, rms_png, rms_csv, oct3_png, oct3_csv, oct24_csv, oct24_png):
        self.rms_png = rms_png
        self.rms_csv = rms_csv
        self.oct3_csv = oct3_csv
        self.oct3_png = oct3_png
        self.oct24_csv = oct24_csv
        self.oct24_png = oct24_png
        self.octavefracs = octavefracs
        return

    def switch_key(self,  key):
        # switch the key of the .sum file to be the same for all .sum file models
        switch_key = {
            1:
            {
                "filename": "original_filename",
                "description": "description",
                "operator": "operator",
                "start_position": "start_position",
                "end_position": "end_position",
                "run_length": "run_length",
                "direction": "direction",
                "rail": "rail",
                "sensor_position": "sensor_position",
                "run_graph_scale": "run_graph_scale",
                "start_date": "start_date",
                "start_time": "start_time",
                "finish_time": "finish_time",
                "sensor_calibration": "sensor_calibration",
                "integrator_time_constant": "integrator_time_constant",
                "long_wavelength_filter": "long_wavelength_filter",
                "encoder_pulse_spacing": "encoder_pulse_spacing",
                "encoder_threshold": "encoder_threshold",
                "sampling_distance": "sampling_distance",
            },
            2:
            {
                "filename": "original_filename",  # tricky
                "description": "description",  # tricky
                "operator": "operator",
                "start_position": "start_position",
                "end_position": "end_position",
                "run_length": "run_length",
                "direction": "direction",  # na
                "rail": "rail",
                "sensor_position": "sensor_position",
                "run_graph_scale": "run_graph_scale",  # na
                "started_at": "start_date",  # tricky
                # "started_at": "start_time",#tricky
                "finished_at": "finish_time",
                "sensor_calibration": "sensor_calibration",
                "integrator_time_constant": "integrator_time_constant",
                "long_wavelength_filter": "long_wavelength_filter",
                "encoder_pulse_spacing": "encoder_pulse_spacing",
                "encoder_threshold": "encoder_threshold",
                "sampling_distance": "sampling_distance",
            }
        }
        return switch_key[self.model][key]

    def parse_sum(self, line):
        parsers = {
            1: self.model_1_parser(line),
            2: self.model_2_parser(line)
        }
        return parsers[self.model]

    def get_sum_model(self, fname):
        f = open(os.path.join(self.root_path, fname), 'r')
        data = f.read()
        if(data.count("=") > 10):
            return 1  # model 1 is '=' separated
        elif(data.count(":") > 10):
            return 2  # model 1 is ':' separated
        else:
            sys.exit(
                "Unknown .sum file type. Add a custom parser for this .sum type")
        return

    def model_1_parser(self, line):
        kv = line.split("=")
        if(len(kv) == 2):
            kv[0] = kv[0].strip().lower().replace(" ", "_")
            kv[1] = kv[1].strip().replace('"', "")
            # special cases

            # rail
            if(kv[0] == 'rail'):
                if(kv[1].lower() == "both"):

                    kv[1] = ['L', 'R']
                elif(kv[1].lower() == 'left'):
                    kv[1] = ['L']
                elif(kv[1].lower() == 'right'):
                    kv[1] = ['R']

            # sensor_position
            if(kv[0] == 'sensor_position'):
                sensor_positions = kv[1].split(";")
                kv[1] = []
                for sp in sensor_positions:
                    kv[1].append(sp.replace("A", "").replace("B", "").replace(
                        "mm", "").replace("-", "").replace(" ", ""))

            # sampling_distance
            if(kv[0] == 'sampling_distance'):
                kv[1] = kv[1].replace('"', '').replace(
                    "mm", "").replace(" ", "")

            try:
                new_key = self.switch_key(kv[0])
            except KeyError:
                print(f"{kv[0]} key is undefined in .sum model.")
                return
            return(new_key, kv[1])
        return

    # TODO : work-in-progress
    def model_2_parser(self, line):
        kv = line.split(":")

        if(len(kv) >= 2):
            kv[0] = kv[0].strip().lower().replace(" ", "_")
            kv[1] = kv[1].strip()
            # rail
            if(kv[0] == 'rail'):

                if(kv[1].lower() == "both"):
                    kv[1] = ['L', 'R']
                elif(kv[1].lower() == 'left'):
                    kv[1] = ['L']
                elif(kv[1].lower() == 'right'):
                    kv[1] = ['R']

            # sensor_position
            if(kv[0] == 'sensor_position'):
                sensor_positions = kv[1].split(";")
                kv[1] = []
                for sp in sensor_positions:
                    kv[1].append(sp.replace("A", "").replace("B", "").replace(
                        "mm", "").replace("-", "").replace(" ", ""))

            # sampling_distance
            if(kv[0] == 'sampling_distance'):
                kv[1] = kv[1].replace('"', '').replace(
                    "mm", "").replace(" ", "")

            # started_at
            if(kv[0] == 'started_at'):
                kv[1] = kv[3].split(" ")[1]

        # description
        if(self.model2_description_tag):
            kv[0] = 'description'
            try:
                kv[1] = line
            except:
                kv.append(line)
            self.model2_description_tag = False

        if("[Run description]" in line):
            self.model2_description_tag = True

        try:
            new_key = self.switch_key(kv[0])
        except KeyError:
            print(f"{kv[0]} key is undefined in .sum model.")
            return
        return(new_key, kv[1])

    def set_summary(self):
        self.file_list = [name for name in os.listdir(
            self.root_path) if os.path.isfile(os.path.join(self.root_path, name))]
        for fname in self.file_list:
            if (".sum" in fname):
                self.model = self.get_sum_model(fname)

                f = open(os.path.join(self.root_path, fname), 'r')
                sum_obj = {}
                sum_obj['filename'] = fname
                for line in f:
                    kv = self.parse_sum(line)
                    if(kv):
                        sum_obj[kv[0]] = kv[1]
                self.sum_obj_arr.append(sum_obj)
        return

    def get_summary(self):
        for sum_obj in self.sum_obj_arr:
            print(sum_obj['filename'])
            for rail, sp in zip(sum_obj['rail'], sum_obj['sensor_position']):
                print(rail, sp, sum_obj['filename'].replace(
                    '.sum', '_'+rail+'.txt'))

            print()
        print(str(len(self.sum_obj_arr)) + ' sum files')
        print(str(len(self.file_list) - len(self.sum_obj_arr)) + ' data files')
        return

    ###################################################
    # BLOCK RMS PROCESSING
    ###################################################

    def block_rms_driver(self, data_fname, dist, disp, plot_title, sampling_distance):
        # Filter 30-100
        filt30_100 = self.filt_process(dist, disp)

        # calc RMS
        x_out, y_out = self.calculate_rms(dist, filt30_100, sampling_distance)
        x_out = x_out.reshape(-1, 1)
        y_out = y_out.reshape(-1, 1)
        rms = np.concatenate((x_out, y_out), axis=1)

        # export RMS csv's and plots
        if(self.rms_csv):
            np.savetxt(os.path.join(self.export_path,
                                    data_fname+"_RMS.csv"), rms, delimiter=",")
        if (self.rms_png):
            self.plot_rms(rms, data_fname, plot_title)
        return

    def filt_process(self, x_in, y_in):
        """
        Args:
        Assumes x_in is in kilometers

        Returns: 
        10-30mm filtered output for y_in
        """
        rows = len(x_in)
        sampdist = abs((x_in[rows-1]-x_in[0]))/(rows-1)
        Ts = sampdist
        Fs = 1/Ts
        Fn = Fs/2
        Rp = 1
        Rs = 50

        freqL = 1/(100e-6)
        freqU = 1/(30e-6)
        Wp = np.array([freqL, freqU])/Fn
        Ws = Wp*np.array([0.8, 1.2])
        [n, Wn] = signal.buttord(Wp, Ws, Rp, Rs)
        sos = signal.butter(n, Wn, btype='bandpass', output='sos')
        w, h = signal.sosfreqz(sos)
        output = signal.sosfiltfilt(sos, y_in)
        return output

    def calculate_rms(self, x_in, y_in, sampling_distance):
        """
        Args:
        Assumes x_in is in kilometers, y_in units dont matter

        Returns: 
        1m Block RMS values and center indices for those values
        """
        sampling_distance = round(sampling_distance, 6)  # classic python math
        num_blocks = np.floor(x_in.size*sampling_distance*1000)
        remainder = x_in.size % int(1e-3/sampling_distance)
        x_split = np.array(np.split(x_in[:-remainder], num_blocks))
        x_rem = x_in[-remainder:]
        y_split = np.array(np.split(y_in[:-remainder], num_blocks))
        y_rem = y_in[-remainder:]
        x_out = np.mean(x_split, axis=1)
        x_rem_out = np.mean(x_rem)
        y_out = np.sqrt(np.mean(y_split**2, axis=1))
        y_rem_out = np.sqrt(np.mean(y_rem**2))
        if(x_rem.size > 500):
            x_out = np.append(x_out, x_rem_out)
            y_out = np.append(y_out, y_rem_out)
        return (x_out, y_out)

    def cqi(self, block_rms_arr):
        return np.quantile(block_rms_arr, 0.95)

    def plot_rms(self, rms, data_fname, plot_title):
        cqi = self.cqi(rms[:, 1])
        plt.figure(figsize=(10, 6))
        plt.plot(rms[:, 0]/1e6, rms[:, 1], color="black")
        plt.plot(rms[:, 0]/1e6, np.ones(len(rms[:, 0]))*cqi, color="blue",
                 label="95th Percentile = " + str(round(cqi, 2)) + " (CQI)")
        plt.ylim([0, 50])
        ax = plt.gca()
        ax.grid()
        ax.set_title("1m Block RMS - " + plot_title)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("1m Block RMS Roughness (microns)")
        ax.legend()
        plt.savefig(os.path.join(self.export_path, data_fname+"_RMS.png"))
        plt.clf()
        plt.close()
        return

    ###################################################
    # OCTAVE SPECTRUM PROCESSING
    ###################################################

    def gen_spectrum_driver(self, data_fname, dist, disp, plot_title):
        """
        Assumes dist is in km
        """
        for octavefrac in self.octavefracs:
            wavelengths, spectrum = self.rough_process(dist, disp, octavefrac)
            octave = np.concatenate((wavelengths, spectrum), axis=1)
            if((octavefrac == 3 and self.oct3_csv) or (octavefrac == 24 and self.oct24_csv)):
                np.savetxt(os.path.join(self.export_path, data_fname +
                                        "_OCT"+str(octavefrac)+".csv"), octave, delimiter=",")
            if((octavefrac == 3 and self.oct3_png) or (octavefrac == 24 and self.oct24_png)):
                self.plot_spectrum(wavelengths, spectrum,
                                   data_fname, octavefrac, plot_title)
        return

    def rough_process(self, dist, rough, octavefrac):
        """
        Args:
        Assumes dist is in kilometers

        Returns: 
        centre wavelengths and spectrum arrays
        """
        dist = dist*1e3  # converts to meters
        rows = len(dist)
        sampdist = abs((dist[rows-1]-dist[0]))/(rows-1)
        totallength = dist[rows-1]-dist[0]
        number = 1/sampdist  # Number of samples required to make up 1m

        if totallength < 2:
            seglength = rows
        else:
            power = 1
            while pow(2, power) < 2*number:
                power += 1
            seglength = pow(2, power)

        if seglength > rows:  # Check that there are enough points in the data to support this seglength
            seglength = seglength/2
            print('Warning: Segment length of less than 1m used')
        Fs = number  # Sampling frequency in samples/m
        # NFFT is the next highest power of 2 greater than or equal to seglength
        NFFT = 2 ** math.ceil(math.log2(seglength))
        fspec, Pxx = signal.welch(  # f - frequencies, Pxx - PSD of signal generated using Welch's method
            rough,
            fs=Fs,
            window='hann',
            nperseg=seglength,
            noverlap=seglength//2,
            nfft=NFFT,
            detrend='linear',
            return_onesided=True,
            scaling='density',
            average='mean'
        )

        centrefreqs, spectrum = self.gen_spectrum(fspec, Pxx, octavefrac)
        wavelengths = np.reciprocal(centrefreqs)*1000
        return wavelengths, spectrum

    def gen_spectrum(self, fspec, Pxx, octavefrac):
        """
        Args:
            fspec: ? 
            Power Spectrum: ?
            OCtave Frac: ?

        Returns: 
        centre frequencies and spectrum arrays
        """
        n = octavefrac  # 1/n octave analysis
        nperband = 1  # For verification of each band, eliminate underrepresented bands
        ratio = pow(10, 0.3/n)

        npts = len(fspec)

        minband = 1  # Arbitrary choices
        maxband = np.fix(15*n)

        # Location of output spectrum values
        spectrum = np.zeros([int(maxband), 1], dtype=float)
        # Number of lines included per band
        nint = np.zeros([int(maxband), 1], dtype=int)

        fcmin = pow(ratio, minband)  # Initial limits
        foctlower = fcmin/math.sqrt(ratio)
        foctupper = fcmin*math.sqrt(ratio)

        iband = 0
        ifreq = 0

        f = fspec
        fmin = fspec[0]
        fmax = fspec[-1]

        # Lower bound for every band
        fnarrowlower = np.concatenate(
            ([fmin], (f[range(1, npts)] + f[range(0, npts-1)])/2))
        # Upper bound for every band
        fnarrowupper = np.concatenate(
            ((f[range(1, npts)] + f[range(0, npts-1)])/2, [fmax]))

        fnarrowlower = np.reshape(fnarrowlower, [len(fnarrowlower), 1])
        fnarrowupper = np.reshape(fnarrowupper, [len(fnarrowupper), 1])

        bwnb = fnarrowupper - fnarrowlower  # Bandwidth for each narrow band

        # Pass through bands until you find first band with data
        while fnarrowlower[ifreq] > foctlower:
            foctlower = foctupper
            foctupper *= ratio
            iband += 1

        # Find first line wholly or partly in this band
        while fnarrowupper[ifreq] < foctlower:
            ifreq += 1

        while foctupper <= fmax:  # Add lines within each band
            # First line overlaps 1/n octave boundary
            if fnarrowupper[ifreq] < foctupper:
                spectrum[iband] += Pxx[ifreq] * \
                    (fnarrowupper[ifreq] - foctlower)
                nint[iband] += 1
                ifreq += 1
            else:  # Line overlaps start and end of 1/n octave band
                spectrum[iband] += Pxx[ifreq] * (foctupper - foctlower)
                nint[iband] += 1

            while fnarrowupper[ifreq] < foctupper:
                spectrum[iband] += Pxx[ifreq] * bwnb[ifreq]
                nint[iband] += 1
                ifreq += 1

            if fnarrowlower[ifreq] > foctlower:  # Last line overlaps 1/n octave boundary
                spectrum[iband] += Pxx[ifreq] * \
                    (foctupper - fnarrowlower[ifreq])
                nint[iband] += 1

            foctlower = foctupper
            foctupper *= ratio
            iband += 1
            if iband >= maxband:
                break

        iband -= 1
        icount = np.array(list(range(1, int(maxband))), ndmin=2).T
        manyratio = np.ones((int(maxband)-1, 1)) * ratio

        centrefreqs = pow(manyratio, icount)

        jband = iband

        while nint[jband] >= nperband:  # Isolate bands with valid data
            jband -= 1
            if jband < 0:
                break

        jband += 1

        spectrum = spectrum[range(jband, iband)]
        centrefreqs = centrefreqs[range(jband, iband)]
        spectrum = 10 * np.log10(spectrum)  # dB

        return centrefreqs, spectrum

    def plot_spectrum(self, wavelengths, spectrum, data_fname, octavefrac, plot_title):
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, spectrum, color="black")
        plt.plot(iso3095Trace_L['x'], iso3095Trace_L['y'],
                 label=iso3095Trace_L['name'], color=iso3095Trace_L['marker']['color'])
        plt.plot(enTrace_L['x'], enTrace_L['y'],
                 label=enTrace_L['name'], color=enTrace_L['marker']['color'])
        plt.plot(grindingAcceptance_L['x'], grindingAcceptance_L['y'],
                 label=grindingAcceptance_L['name'], color=grindingAcceptance_L['marker']['color'])
        plt.plot(grindingPolishing_L['x'], grindingPolishing_L['y'],
                 label=grindingPolishing_L['name'], color=grindingPolishing_L['marker']['color'])
        plt.xscale('log')
        ax = plt.gca()
        ax.invert_xaxis()
        ax.set_xticks(iso3095Trace_L['x'])
        ax.set_xticklabels(iso3095Trace_L['x'], rotation=45)
        ax.legend()
        ax.set_xlabel("Octave Band Centre Wavelength")
        ax.set_ylabel("Roughness (db rel 1 micron)")
        if(octavefrac == 3):
            ax.set_title("3rd Octave - " + plot_title)
        else:
            ax.set_title("24th Octave - " + plot_title)
        ax.set_ylim([-20, 40])
        ax.grid()
        plt.savefig(os.path.join(self.export_path,
                                 data_fname+"_OCT"+str(octavefrac)+".png"))
        plt.close()
        plt.clf()
        return
