import xml.etree.ElementTree as ET

label = ET.parse("../coffeebeans_label/Set04-good.01.01.xml")
root = label.getroot()
for neighbor in root.iter('path'):
    print(neighbor.attrib)