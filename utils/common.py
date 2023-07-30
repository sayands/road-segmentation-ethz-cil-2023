import json
import os
import os.path as osp
import pickle


def ensure_dir(path):
    """
    Creates the given path if it doesn't exist
    @param path: path to be created if non-existent
    """
    if not osp.exists(path):
        os.makedirs(path)


def assert_dir(path):
    """
    asserts that path exists
    @param path: path
    """
    assert osp.exists(path)


def load_pkl_data(filename):
    """
    Load pickle object from the given file
    @param filename: path to pickle file
    @return:
    """
    with open(filename, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict


def write_pkl_data(data_dict, filename):
    """
    Saves the given data to a pickle file
    @param data_dict: data to be saved
    @param filename: location where data_dict will be saved
    """
    with open(filename, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    """
    Loads JSON file
    @param filename: json_file location
    @return: the loaded json as dictionary
    """
    file = open(filename)
    data = json.load(file)
    file.close()
    return data


def write_json(data_dict, filename):
    """
    Writes the data_dict to the given filename
    @param data_dict: data to be serialized
    @param filename: path to the location where data will be saved
    """
    json_obj = json.dumps(data_dict, indent=4)

    with open(filename, "w") as outfile:
        outfile.write(json_obj)


def get_print_format(value):
    """
    Return's the correct print format for the type of value
    @param value: the value to be printed
    @return:corresponding string format
    """
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.6f'


def get_format_strings(kv_pairs):
    """
    Get format string for a list of key-value pairs.
    @param kv_pairs: the pairs for which we want the format strings
    @return:format_strings
    """
    log_strings = []
    for key, value in kv_pairs:
        fmt = get_print_format(value)
        format_string = '{}: {:' + fmt + '}'
        log_strings.append(format_string.format(key, value))
    return log_strings


def update_dict(dictionary, to_add_dict):
    for key in dictionary.keys():
        dictionary[key].append(to_add_dict[key])
    return dictionary


def get_log_string(result_dict, name=None, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None,
                   timer=None):
    """
    Prepares string to bn logged
    @param result_dict: dictionary with information to be logged
    @param name: leading name of the log
    @param epoch: epoch to which the log is corresponding
    @param max_epoch: the total number of epochs
    @param iteration: at which iteration the log is done
    @param max_iteration: the total number of iterations
    @param lr: learning rate
    @param timer: timer for the time to be logged
    @return: constructed message to be logged
    """
    log_strings = []
    if name is not None:
        log_strings.append(name)
    if epoch is not None:
        epoch_string = f'Epoch: {epoch}'
        if max_epoch is not None:
            epoch_string += f'/{max_epoch}'
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f'iter: {iteration}'
        if max_iteration is not None:
            iter_string += f'/{max_iteration}'
        if epoch is None:
            iter_string = iter_string.capitalize()
        log_strings.append(iter_string)
    if 'metadata' in result_dict:
        log_strings += result_dict['metadata']
    for key, value in result_dict.items():
        if key != 'metadata':
            format_string = '{}: {:' + get_print_format(value) + '}'
            log_strings.append(format_string.format(key, value))
    if lr is not None:
        log_strings.append('lr: {:.3e}'.format(lr))
    if timer is not None:
        log_strings.append(timer.tostring())

    message = ', '.join(log_strings)
    return message
