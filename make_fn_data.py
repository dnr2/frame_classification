from __future__ import print_function

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

#TODO: Change this to your own directory
FN_FOLDER = "C:\\Users\\danil\\Documents\\Northwestern\\QRG\\Rep\\ea\\FrameNet_1.7\\"
FN_LU_FOLDER = FN_FOLDER + "lu\\"

DOMAINS = {"fn": "http://framenet.icsi.berkeley.edu"}

def load_fn_data():
	"""
	Returns a JSON like object with lex units data
	"""

	lu_dir_entries = listdir(FN_LU_FOLDER)

	lex_units = []

	for lu_dir_entry in lu_dir_entries:

		lu_file = join(FN_LU_FOLDER, lu_dir_entry) 
		if isfile(lu_file) and lu_file.endswith(".xml"):
			
			lex_unit = {}
			element = ET.parse(lu_file)
			root = element.getroot()

			lex_unit["file_name"] = lu_file
			lex_unit["frame"] = root.attrib["frame"]
			lex_unit["frame_id"] = root.attrib["frameID"]
			lex_unit["lex_unit"] = root.attrib["name"]
			lex_unit["pos"] = root.attrib["POS"]
			lex_unit["definition"] = element.find("./fn:definition", DOMAINS).text
			lex_unit["lexeme"] = root.find("./fn:lexeme", DOMAINS).attrib["name"]
			lex_unit["sentences"] = []

			for sentence_el in element.findall("./fn:subCorpus/fn:sentence", DOMAINS):
				sentence = {}

				targets = sentence_el.findall(
					"./fn:annotationSet/*[@name='Target']/fn:label", DOMAINS)
				indexes = []
				for target in targets:
					start = target.attrib["start"]
					end = target.attrib["end"]
					indexes.append((start, end))

				sentence["text"] = sentence_el.find("./fn:text", DOMAINS).text
				sentence["indexes"] = indexes
				lex_unit["sentences"].append(sentence)

			lex_units.append(lex_unit)

	print("statistics")
	print("# lex units: ", len(lex_units))
	print("# frames: ", len(set([x["frame"] for x in lex_units])))
	print("# data points: ", len([y for x in lex_units for y in x["sentences"]]))
	print("# lex units without data: ", len([x for x in lex_units if len(x["sentences"]) == 0]))

	return lex_units

if __name__=="__main__": 
    load_fn_data()