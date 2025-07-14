from scipy.io import loadmat


def load_mat_file(file_path, debug=False):
    """
    Load a .mat file and return its contents.

    Parameters:
    file_path (str): The path to the .mat file.

    Returns:
    dict: A dictionary containing the contents of the .mat file.
    """
    try:
        data = loadmat(file_path)
        if debug:
            print("################## for debug ################")
            print(f"keys: {data.keys()}")
            print("################## for debug ################")
        return data
    except Exception as e:
        print(f"An error occurred while loading the .mat file: {e}")
        return None


data = load_mat_file(
    "/fast_storage/hyeokgi/data_v2_slice_512/train/54523548_3_slice_16.mat"
)
print(data)
if data is not None:
    print(data.keys())
