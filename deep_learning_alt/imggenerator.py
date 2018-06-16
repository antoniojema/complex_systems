from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import subprocess
import numpy as np

fonts = (subprocess.Popen(['fc-list'],stdout=subprocess.PIPE).communicate())[0]

font_str = []
goon = True
n = 0
while goon:
	n+=1
	print n
	i=0
	goon2=True
	while goon2:
		if fonts[i] == ':' or fonts[i] == '\npyth':
			goon2 = False
		elif i < len(fonts)-2:
			i += 1
		else:
			goon2 = False
			goon = False
	font_str += [fonts[0:i]]
	print fonts[0:i]
	#fonts = fonts - fonts[0:i+2]
	fonts = fonts[i:]
	
	i=0
	goon2=True
	while goon2:
		if fonts[i] == '\n':
			goon2 = False
		elif i < len(fonts)-2:
			i += 1
		else:
			goon2 = False
			goon = False
	print fonts[0:i+1]
	#fonts = fonts - fonts[0:i+1]
	fonts = fonts[i+1:]
	
#print np.array(font_str)

for i in range(len(font_str)):
	font = ImageFont.truetype(font_str[i],80)
	for j in '0123456789':
		img = Image.new('RGB', (100,100),(255,255,255))
		draw = ImageDraw.Draw(img)
		pos = (100 - draw.textsize(j,font=font)[0]) / 2
		draw.text((pos, 0),j,(0,0,0),font=font)
		#img.show()
		img.save('data/img'+str(j)+'{0:03}'.format(i)+'.bmp')

