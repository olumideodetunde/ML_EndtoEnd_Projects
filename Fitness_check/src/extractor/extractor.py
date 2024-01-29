import xml.etree.ElementTree as ET
class Extractor:
    '''
    This class extracts desired data points from an xml file.
    '''

    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.records = None

    def get_records(self, element_name:str) -> list:
        '''
        Args: element_name (str): name of the element to extract from the xml file

        This method extracts all the records of a given element from the xml file.

        Returns: list of dictionaries      
        '''
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

    def extract_datapoints(self, desired_dataname:list) -> list:
        '''
        Args: desired_datavalues (list): list of desired data values to extract from the xml file
    
        This method extract the specific data points from the dictiary list of defined records
        using the data name (keys).
    
        Returns: list of dictionaries
        '''
        data_points = []
        for y in self.records:
            workout_detail =  y.get('attributes')
            unfiltered = y.get('children')
            filtered = [d for d in unfiltered if isinstance(d, dict)]
            for d in filtered:
                if d.get('type') in desired_dataname:
                    workout_detail.update(d)
            data_points.append(workout_detail)
        return data_points