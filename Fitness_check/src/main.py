'''This module extracts desired data points from a defined 
    xml file element and saves to a csv file.'''
import logging
import argparse
import pandas as pd
from extractor.extractor import Extractor
def main():
    '''
    This script extracts desired data points from a defined xml file 
    element and saves to a csv file.
    '''
    parser = argparse.ArgumentParser(description="Extract data points from an xml file and \
                                        save to a csv file.")
    parser.add_argument("--input_path", type=str, help="Path to the input xml file.")
    parser.add_argument("--record_name", type=str, help="Name of the record_element to \
                        extract from the xml file.")
    parser.add_argument("--desired_dataname_list", nargs="+", type=str,  help="List of desired \
                        data names to extract from the record_element.")
    parser.add_argument("--output_path", type=str, help="Path to the output csv file.")
    args = parser.parse_args()
    input_path = args.input_path
    record_name = args.record_name
    desired_dataname_list = args.desired_dataname_list
    output_path = args.output_path

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    x = Extractor(input_path)
    x.get_records(record_name)
    data_points = x.extract_datapoints(desired_dataname_list)
    data_points_df = pd.DataFrame(data_points)
    data_points_df.to_csv(output_path)
    logging.info(f"Data points extracted from {input_path} and saved to {output_path}")
if __name__ == "__main__":
    main()
#End of file