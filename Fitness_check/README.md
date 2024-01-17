# Project Title

Fitness Check Extractor

## Description

This program extracts desired datapoints from health data exported as XML from the Apple Health app. The program is written in Python and uses the ElementTree library to parse the XML file. 

## Getting Started

### Dependencies

* see requirements.txt

### Installing

* git clone the parent repository ML_end_to_end_projects to your local machine
* navigate to the Fitness_check folder 
* create a data folder called data in the Fitness_check folder
* download the XML file from the Apple Health app and save it in the data folder
* install the required packages using the requirements.txt file by running the following command in the terminal:

```
pip install -r requirements.txt
```

### Executing program

* ensure the terminal is in the Fitness_check folder
* run the following command in the terminal:
```
python fitness_check.py --input_file data/your_xml_file.xml --record_name your_record_name --desired_datanames your_desired_datanames --output_file data/your_output_file.csv

```

## Authors

Contributors names and contact info
Olumide Odetunde [@OlumideOdetunde](https://www.linkedin.com/in/olumide-odetunde/)
