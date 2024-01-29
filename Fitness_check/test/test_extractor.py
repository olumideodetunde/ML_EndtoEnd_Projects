'''This is the test for the Extractor class'''
#%%
import pytest
from src.extractor.extractor import Extractor

@pytest.fixture
def sample_xml(tmp_path):

    xml_content = '''
    <Record type="HKQuantityTypeIdentifierBodyMassIndex" sourceName="MyNetDiary" sourceVersion="223" unit="count" 
    creationDate="2023-03-31 07:28:52 +0000" startDate="2023-03-31 07:28:52 +0000" endDate="2023-03-31 
    07:28:52 +0000" value="30.5">
    <MetadataEntry key="HKWasUserEntered" value="1"/>
    <MetadataEntry key="timeEstimate" value="1"/>
    </Record>
    '''

    xml_file = tmp_path / 'sample.xml'
    xml_file.write_text(xml_content)
    return xml_file

class TestExtractor:
    def test_get_records(self, sample_xml):
        t_extractor = Extractor(str(sample_xml))
        parameter = 'Workout'
        result = Extractor.get_records(t_extractor, parameter)
        assert isinstance(result, list)