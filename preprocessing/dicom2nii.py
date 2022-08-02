import pydicom
import os
import SimpleITK as sitk
import numpy as np


def dicom2arr(dcm_directory):
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_directory)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    try:
        image3D = series_reader.Execute()
        image_array = sitk.GetArrayFromImage(image3D)
    except:
        print('error:' + dcm_directory)
        image_array = 'err'
    return image_array


def get_protocol_name(path):
    dcm_list = os.listdir(path)
    dcm_file = os.path.join(path, dcm_list[0])
    header = pydicom.dcmread(dcm_file)
    try:
        protocol_name = header.ProtocolName
    except:
        print('protocol_error:' + path)
        protocol_name = 'no_name'
    return protocol_name


root = '/DICOM'
target_root = '/ori'
patients_list = os.listdir(root)

for patient in patients_list:
    patient_path = os.path.join(root, patient)
    print(patient)
    for path, dir_list, file_list in os.walk(patient_path):
        if len(file_list) > 5:
            print(path)
            phase_arr = dicom2arr(path)
            if not phase_arr == 'err':
                protocol_name = get_protocol_name(path).replace('/', '_')
                out = sitk.GetImageFromArray(phase_arr)
                out_name = os.path.join(target_root, patient,
                                        protocol_name + '.nii')
                patient_folder = os.path.split(out_name)[0]
                if not os.path.exists(patient_folder): os.mkdir(patient_folder)
                while os.path.exists(out_name):
                    out_name = os.path.join(target_root, patient,
                                            protocol_name + str(np.random.randint(10000)) + '.nii')
                sitk.WriteImage(out, out_name)
