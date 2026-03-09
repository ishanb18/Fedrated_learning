##### UC Merced Land Use

A balanced aerial scene classification benchmark with 21 land-use classes; ideal for evaluating BoVW/CNN baselines but less diverse than multi-source datasets, so domain adaptation is often needed.



Owner/maintainer: UC Merced (Shawn Newsam lab); cite Yang \& Newsam 2010.



Task type: Scene classification (21-way).



Number of classes: 21 (agricultural, airplane, baseball diamond, beach, buildings, chaparral, dense residential, forest, freeway, golf course, harbor, intersection, medium residential, mobile home park, overpass, parking lot, river, runway, sparse residential, storage tanks, tennis court).



Images per class: 100 (balanced).



Total images: 2,100.



Image size: 256×256, RGB.



Ground sample distance: ≈1 ft (≈0.3 m) per pixel.



Features: filename (text), image (H×W×3, uint8), label (21-class).



Splits: Commonly user-defined; many releases provide a single 2,100 example pool.



##### Aerial Image Dataset (AID)

A labeled aerial scene classification benchmark of 10,000 Google Earth RGB images across 30 categories, curated over varied regions/seasons/sensors to encourage domain-robust scene features; harder than single-source datasets like UC Merced due to multi-domain variability.



Owner/maintainer: Xia et al., Wuhan University; cite AID (IEEE TGRS 2017).



Task type: Scene classification (30-way).



Number of classes: 30 (airport, bare land, baseball field, beach, bridge, center, church, commercial, dense residential, desert, farmland, forest, industrial, meadow, medium residential, mountain, park, parking, playground, pond, port, railway station, resort, river, school, sparse residential, square, stadium, storage tanks, viaduct).



Images per class: ≈200–400.



Total images: 10,000.



Image size: 600×600, RGB.



Source: Multi-source Google Earth across diverse locations/times/conditions.



##### DeepGlobe

CVPR 2018 challenge with three tracks—land cover segmentation (7 classes), building detection, and road extraction—on high-resolution satellite tiles with expert annotations and fixed train/val/test splits for fair benchmarking.

###### 

###### Landcover



Task: Semantic segmentation into 7 classes (agriculture\_land, urban\_land, rangeland, water, barren\_land, forest\_land, unknown).



Total images: 1,146 (train 803, val 171, test 172); class distribution is pixel-wise and skewed.



Image size/resolution: Typically 2448×2448 RGB at ~50 cm GSD.



Annotation: Pixel masks; some tiles unlabeled by design.



###### Road extraction



Task: Binary semantic segmentation (road vs. background).



Total images: 8,570 tiles (~2,220 km²).



Splits: Train 6,226; Val 1,243; Test 1,101 (~70/15/15).



Class balance: Road pixels ≈4.5% (train), 3% (val), 4.1% (test).



Resolution/GSD: ~50 cm GSD RGB; tiled patches with binary masks.



###### Building detection



Classes: 2 (building vs. background), instance-level evaluation (IoU≥0.5, F1).



Imagery/annotations: High‑res RGB (DigitalGlobe/Maxar); polygon footprints/masks, small fragments typically ignored in scoring.



Splits/counts: Train/val/test provided in the challenge bundle; exact image counts vary by package/mirror.

