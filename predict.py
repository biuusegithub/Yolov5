from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()

    crop = False
    count = False

    img1 = "img/street.jpg"
    img2 = "img/fruit_and_person.jpg"

    img_list = []
    img_list.append(img1)
    img_list.append(img2)
 

    for img in img_list:
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo(image, crop=crop, count=count)
            r_image.show()

