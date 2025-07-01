0. Die Daten vervollständigen (leere = gleiche Position)
1. Grafika pro Benutzer mit Pfad
2. Timestamp Seasonality - heatmap
   Tag 1 2 3 4 5 6 ...
   1 Geobleu(tag1, tag4)
   2
   3
   4
   5
   6
   7
   ...

Tag = 48 timestamps

Danach - DBSCAN / dichtenbasiertes Clustering

Benutzer 1 ... 1000
1
...
1000

3.  16.06.

1.  Für die vollständigen die 14 Tage abschneiden, mit baseline Methods füllen und mit geobleu evaluieren
1.  GeoBleu Heatmap machen - für Benutzer pro Tag und für Tag pro Benutzer
1.

pool Benutzer - vollständig
pool Benutzer - to_predict
u \in to_predict ->
auf Tage -> timestamps in Gruppen von 48 gruppieren

u0 -> [t1, t2, ..., t60]

geobleu(u0, u1), geobleu(u0, u2), ..., geobleu(u0, un)

u1 -> [t1, t2, ..., t60, ..., t75]
u2 -> [t1, t2, ..., t60, ..., t75]
...
un -> [t1, t2, ..., t60, ..., t75]

1 3 7 19

200x200

160000 Varianten pro Benutzer - shifts / offsets x y -200..+200

start -> relative movement -> geobleu evaluieren -> with relative vervollständigen

4 6 10 22

2. mit gaussian top_n kandidaten evalieren

## 28.05

### TODOs:

Expose hinweise dazuhinzf ins Englisch uebersetzen und beides an 4
Naechste Woche Termin machen
Visualisierung
Basic benchmarks reproduzieren
Code fuer Paper schreiben

top-1

top-5
rel_path_top1 = rel_base1 + rel_ext1
rel_path_top2 = rel_base2 + rel_ext2
rel_path_top3 = rel_base3 + rel_ext3
rel_path_top4 = rel_base4 + rel_ext4
rel_path_top5 = rel_base5 + rel_ext5

rel_ext1 = (xd1, yd1), (xd2, yd2), (xd3, yd3), (xd4, yd4), (xd5, yd5), ...
rel_ext2 = (xd1, yd1), (xd2, yd2), (xd3, yd3), (xd4, yd4), (xd5, yd5), ...
rel_ext3 = (xd1, yd1), (xd2, yd2), (xd3, yd3), (xd4, yd4), (xd5, yd5), ...

|
\/

random_walks

rel_ext_q = (xd1, yd1), (xd2, yd2), (xd3, yd3), (xd4, yd4), (xd5, yd5), ...

mean
mode
