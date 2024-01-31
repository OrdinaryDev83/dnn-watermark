pita-default-config:
	python3 pita.py generate --size 30 --split test
	python3 pita.py generate --size 20 --split val
	python3 pita.py generate --size 100 --split train

pita-yolo-config:
	python3 pita.py generate --size 1 --split test --format yolo
	python3 pita.py generate --size 1 --split val --format yolo
	python3 pita.py generate --size 1 --split train --format yolo

clean:
	rm -rf data/
	rm -rf output_dataset/
