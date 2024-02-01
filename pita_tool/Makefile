pita-default-config:
	python3 pita.py generate --size 12800 --format coco --split train --images-path unlabeled2017/
	python3 pita.py generate --size 4000 --format coco --split val --images-path unlabeled2017/
	python3 pita.py generate --size 3200 --format coco --split test --images-path unlabeled2017/

pita-yolo-config:
	python3 pita.py generate --size 12800 --format yolo --split train --images-path unlabeled2017/
	python3 pita.py generate --size 4000 --format yolo --split val --images-path unlabeled2017/
	python3 pita.py generate --size 3200 --format yolo --split test --images-path unlabeled2017/

clean:
	rm -rf data/
	rm -rf output_dataset/
