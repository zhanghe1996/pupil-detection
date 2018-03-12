import os
import json
import urllib2

if __name__ == "__main__":
	with open('EyeSnap_2017_v10.json') as json_file:
		data = json.load(json_file)

	gazes = {'FORWARD_GAZE', 'LEFTWARD_GAZE', 'RIGHTWARD_GAZE', 'UPWARD_GAZE'}

	image_path = os.path.join('..', 'eye_test', 'data', 'Images')

	for case in data['results']:
		for gaze in gazes:
			if gaze in case:
				url = case[gaze]['url']
				req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"})
				con = urllib2.urlopen(req)
				with open(os.path.join(image_path, case[gaze]['name'] + '.jpeg'), 'wb') as f:
					f.write(con.read())