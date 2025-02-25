import cv2
from detect_constellation import detection
from star_detection import StarDetection

def draw_constellation_connections(img, constellation, stars_within_bbox):
    # tbd: right now it doesn't work cause we don't detect all stars and the order I chose is wrong
    # another idea of order: sort by distance(or maybe angles) to the center of box
    CONNECTION_PATTERNS = {
        'orion': [
            (0,3),(1,6),(2,6),
            (3,5),(4,6),(4,7),
            (5,8),(8,9),(5,7),
            (6,13),(7,10),(10,11),
            (11,13),(13,15),(10,12),
            (12,14),(14,15)
        ],
        'aquila': [
            (0,2),(1,2),(2,3),
            (3,6),(2,4),(5,6),
            (4,5),(0,4),(4,7)
        ],
        'bootes': [
            (0,1),(0,2),(1,3),
            (2,4),(3,5),(4,5),
            (5,6),(5,7)
        ],
        'canis_major': [
            (0,1),(0,2),(1,2),
            (2,3),(3,4),(4,5),
            (5,6),(5,7),(7,11),
            (11,9),(9,8),(8,10),
            (8,12),(3,8),(11,13),
            (13,14),(13,15)        
        ],
        'canis_minor': [
            (0,1)
        ],
        'cassiopeia': [
            (0,2),(2,1),(1,4),(4,3)
        ],
        'cygnus' : [
            (0,1),(1,3),(3,4),
            (4,2),(4,6),(6,8),
            (8,7),(5,9),(4,5)
        ],
        'gemini' : [
            (0,1),(0,2),(2,4),
            (4,7),(7,8),(8,9),
            (9,10),(1,3),(3,5),(5,6)
        ],
        'leo' : [
            (1,2),(1,3),(3,4),
            (4,5),(4,6),(6,9),
            (9,8),(8,7),(7,5)
        ],
        'lyra' : [
            (0,1),(1,2),(0,2),
            (2,3),(2,4),(4,5),(5,3)
        ],
        'sagittarius' : [
            (1,2),(2,3),(3,4),
            (4,12),(12,13),(12,8),
            (8,7),(7,5),(5,0),
            (5,9),(9,10),(10,6),
            (9,14),(14,16),(11,12),
            (11,15),(15,17),(17,18),
            (18,20),(20,21),(20,21)
        ],
        'scorpius' : [
            (0,2),(1,2),(2,3),
            (2,5),(3,4),(4,6),
            (6,11),(11,10),(10,9),
            (9,8),(8,7)
        ],
        'taurus' : [
            (0,1),(1,3),(3,5),
            (5,4),(4,2),(5,7),
            (7,6),(6,8),(8,10),
            (7,9),(9,11)
        ],
        'ursa_major' : [
            (0,1),(1,2),(2,3),
            (3,4),(3,5),(4,7),
            (5,7),(5,10),(10,12),
            (12,16),(12,17),(4,6),
            (6,8),(8,9),(9,11),
            (11,7),(11,13),(13,14),(13,15)
        ]

    }

    class_name = constellation['class_name']
    connections = CONNECTION_PATTERNS.get(class_name, [])

    if not connections:
        print(f"No connection pattern defined for {class_name}.")
        return

    for (sx, sy) in stars_within_bbox:
        cv2.circle(img, (sx, sy), 3, (0, 255, 255), -1)

    for (start_idx, end_idx) in connections:
        if start_idx < len(stars_within_bbox) and end_idx < len(stars_within_bbox):
            start_point = (stars_within_bbox[start_idx][0], stars_within_bbox[start_idx][1])
            end_point = (stars_within_bbox[end_idx][0], stars_within_bbox[end_idx][1])
            cv2.line(img, start_point, end_point, (0, 255, 255), 2)


def map_stars_to_constellations(img, detected_constellations, all_stars):
    colors = {
        'orion': (227, 252, 3),
        'aquila': (152, 252, 3),
        'bootes' : (3, 252, 231),
        'canis_major' : (3, 194, 252),
        'canis_minor' : (111, 3, 252),
        'cassiopeia' : (190, 3, 252),
        'cygnus' : (252, 3, 186),
        'gemini' : (95, 173, 245),
        'leo' : (252, 111, 3),
        'lyra' : (252, 244, 3),
        'saggitarius' : (3, 252, 11),
        'scorpius' : (83, 158, 58),
        'taurus' : (58, 158, 150),
        'ursa_major': (158, 58, 123),
        'moon' : (226, 235, 164),
        'pleiades' : (184, 29, 29)
    }

    constellations_with_stars = []
    img = cv2.imread(img)

    for constellation in detected_constellations:
        x1, y1, x2, y2 = constellation['bbox']
        cv2.putText(img, f"{constellation['class_name']}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[constellation['class_name']], 1)

        stars_in_bbox = []

        for star in all_stars:
            sx, sy = star
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                stars_in_bbox.append(star)
                cv2.circle(img, (sx, sy), 2, colors[constellation['class_name']], 2)

        stars_in_bbox_sorted = sorted(stars_in_bbox, key=lambda s: s[0]) # we might need them in order
        # draw_constellation_connections(img, constellation, stars_in_bbox_sorted)
        constellations_with_stars.append({
            'constellation': constellation,
            'stars': stars_in_bbox_sorted
        })

    cv2.imshow("Detected constellation stars", img)
    cv2.waitKey(0)
    return constellations_with_stars


if __name__ == "__main__":
    image_to_solve = "C:/Users/Andreea/Documents/constellation/Star-detection/dataset/test/images/2022-01-02-00-00-00-s_png_jpg.rf.da902dcc3763024472a80ca077612fcc.jpg"
    image_to_solve = "C:/Users/Andreea/Documents/constellation/Star-detection/dataset/test/images/2022-01-09-00-00-00-s_png_jpg.rf.1b4788ef2a761e6133a58192102c6160.jpg"
    image_to_solve = "C:/Users/Andreea/Documents/constellation/Star-detection/dataset/test/images/2022-01-11-00-00-00-s_png_jpg.rf.3b967c1738b7800202be12fc4fc19203.jpg"
    image_to_solve = "C:/Users/Andreea/Documents/constellation/Star-detection/dataset/test/images/2022-02-03-00-00-00-n_png_jpg.rf.249728aa17c0b712f336066d9595343b.jpg"
    image_to_solve = "C:/Users/Andreea/Documents/constellation/Star-detection/dataset/test/images/2022-01-11-00-00-00-s_png_jpg.rf.3b967c1738b7800202be12fc4fc19203.jpg"
    detect = StarDetection(image_to_solve)
    stars = detect.blob_detection1()
    constellations = detection(image_to_solve)
    map_stars_to_constellations(image_to_solve, constellations, stars)
