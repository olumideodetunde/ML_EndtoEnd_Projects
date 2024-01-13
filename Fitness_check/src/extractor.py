#%%
import pandas as pd
import xml.etree.ElementTree as ET

class Extractor:

    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.records = None

    def get_records(self, element_name):
        stack = self.root.findall(element_name)
        self.records = []
        while stack:
            record = {}
            current_element = stack.pop(0)
            if list(current_element):
                record['attributes'] = current_element.attrib
                record['children'] = [child.tag for child in current_element]
                for child in current_element:
                    record['children'].append(child.attrib)
            self.records.append(record)
        return self.records

    def extract_desired_datapoints(self, desired_datavalues=None):
        data_points = []
        for y in self.records:
            workout_detail =  y.get('attributes')
            unfiltered = y.get('children')
            print(workout_detail)
            for d in unfiltered:
                if  isinstance(d, dict) and d.get('type') in desired_datavalues:
                    workout_detail.update(d)
                data_points.append(workout_detail)
        return data_points

#%%
if __name__ == '__main__':
    
    file_path = "data/input/apple_health_export/export.xml"
    extractor = Extractor(file_path)
    extractor.get_records('Workout')
    extractor.extract_desired_datapoints(desired_datavalues=['HKQuantityTypeIdentifierDistanceWalkingRunning','HKQuantityTypeIdentifierDistanceCycling'])
    # x = extractor.extract_desired_datapoints(desired_datavalues=['HKQuantityTypeIdentifierDistanceWalkingRunning','HKQuantityTypeIdentifierDistanceCycling'])
    # x_df = pd.DataFrame(x)
    # x_df
    #x_df = x.convert_to_df(x) 
            


# %%
