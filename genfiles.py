
# write train file (just the image list)
import os

with open('C:\\datasets\\bcdd\\train.txt', 'w') as out:
    for img in [f for f in os.listdir('C:\\datasets\\bcdd\\train') if f.endswith('jpg')]:
        out.write('C:\\datasets\\bcdd\\train\\' + img + '\n')

# write the valid file (just the image list)
import os

with open('C:\\datasets\\bcdd\\valid.txt', 'w') as out:
    for img in [f for f in os.listdir('c:\\datasets\\bcdd\\valid') if f.endswith('jpg')]:
        out.write('C:\\datasets\\bcdd\\valid\\' + img + '\n')