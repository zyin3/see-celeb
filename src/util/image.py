import base64
import struct
import os


def readline(line):
  MID, EntityNameString, _, FaceID, FaceRectangle, FaceData = line.split("\t")
  rect = struct.unpack("ffff", base64.b64decode(FaceRectangle))
  return MID, EntityNameString, FaceID, rect, base64.b64decode(FaceData)


def writeImage(filename, data):
  with open(filename, "wb") as f:
    f.write(data)


def unpack(filename, target="img"):
  i = 0
  with open(filename, "r") as f:
    for line in f:
      MID, _, FaceID, FaceRectangle, FaceData = readline(line)
      img_dir = os.path.join(target, MID)
      if not os.path.exists(img_dir):
        os.mkdir(img_dir)
      img_name = "%s" % (FaceID) + ".jpg"
      writeImage(os.path.join(img_dir, img_name), FaceData)
      i += 1
      if i % 1000 == 0:
        print(i, "imgs finished")
  print("all finished")


filename = "/usr/local/google/home/zyin/Downloads/MsCelebV1-Faces-Cropped-DevSet1.tsv"
unpack(filename)
