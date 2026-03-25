#!/bin/bash
ELKI="java -jar elki-bundle-0.8.1-SNAPSHOT.jar"

$ELKI greedyensemble.ComputeKNNOutlierScores \
-dbc.in 6-gaussian-4d.csv \
-krange 10 \
-app.out reference-outlier-scores.csv

cat - >> reference-outlier-scores.csv << EOF
ABOD-poly2 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.anglebased.ABOD \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
LBABOD-10-poly2 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.anglebased.LBABOD \
-fastabod.k 10 -abod.l 10 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
LID-20-Hill $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.intrinsic.LID \
-id.k 20 -id.estimator HillEstimator \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
SOS-4.5 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.distance.SOS \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
ISOS-20-Hill $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.intrinsic.ISOS \
-isos.k 20 -isos.estimator HillEstimator \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
IDOS-20-Hill $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.intrinsic.IDOS \
-idos.kc 20 -idos.kr 20 -idos.estimator HillEstimator \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
ALOCI-10 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.lof.ALOCI \
-loci.nmin 10 \
-loci.seed 0 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
LOCI-r0.2 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.lof.LOCI \
-loci.rmax 0.2 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
IForest-full $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.density.IsolationForest \
-iforest.numtrees 100 \
-iforest.subsample 200 \
-iforest.seed 0 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
COP-10 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.COP \
-cop.k 10 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
SOD-10 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.subspace.SOD \
-sharedNearestNeighbors 10 -sod.knn 10 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
DBOD-10 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.distance.DBOutlierDetection \
-dbod.d 0.25 -dbod.p 0.95 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
DBOS-10 $( $ELKI cli \
-dbc.in 6-gaussian-4d.csv \
-algorithm outlier.distance.DBOutlierScore \
-dbod.d 0.25 \
-evaluator NoAutomaticEvaluation \
-resulthandler tutorial.outlier.SimpleScoreDumper | cut -f2 -d" " | paste -sd" " )
EOF
