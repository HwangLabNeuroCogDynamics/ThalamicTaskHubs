conditions=(body-others faces-otherss places-others tools-others)

coef=([26] [29] [32] [35])

tstat=([27] [30] [33] [36])

for c in ${!conditions[@]}
do
3dMEMA -prefix /data/backed_up/shared/xitchen_WM/stats/MEMA_GS/MEMA_${conditions[$c]}_GS \
-set ${conditions[$c]} \
100307 /data/backed_up/shared/xitchen_WM/results/100307/100307_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/100307/100307_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
100408 /data/backed_up/shared/xitchen_WM/results/100408/100408_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/100408/100408_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
100610 /data/backed_up/shared/xitchen_WM/results/100610/100610_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/100610/100610_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
101006 /data/backed_up/shared/xitchen_WM/results/101006/101006_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/101006/101006_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
101107 /data/backed_up/shared/xitchen_WM/results/101107/101107_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/101107/101107_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
101309 /data/backed_up/shared/xitchen_WM/results/101309/101309_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/101309/101309_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
101915 /data/backed_up/shared/xitchen_WM/results/101915/101915_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/101915/101915_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
102311 /data/backed_up/shared/xitchen_WM/results/102311/102311_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/102311/102311_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
102513 /data/backed_up/shared/xitchen_WM/results/102513/102513_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/102513/102513_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
103111 /data/backed_up/shared/xitchen_WM/results/103111/103111_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/103111/103111_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
103414 /data/backed_up/shared/xitchen_WM/results/103414/103414_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/103414/103414_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
103515 /data/backed_up/shared/xitchen_WM/results/103515/103515_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/103515/103515_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
103818 /data/backed_up/shared/xitchen_WM/results/103818/103818_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/103818/103818_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
105014 /data/backed_up/shared/xitchen_WM/results/105014/105014_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/105014/105014_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
105115 /data/backed_up/shared/xitchen_WM/results/105115/105115_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/105115/105115_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
106016 /data/backed_up/shared/xitchen_WM/results/106016/106016_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/106016/106016_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
107321 /data/backed_up/shared/xitchen_WM/results/107321/107321_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/107321/107321_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
107422 /data/backed_up/shared/xitchen_WM/results/107422/107422_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/107422/107422_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
108121 /data/backed_up/shared/xitchen_WM/results/108121/108121_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/108121/108121_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
108222 /data/backed_up/shared/xitchen_WM/results/108222/108222_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/108222/108222_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
108828 /data/backed_up/shared/xitchen_WM/results/108828/108828_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/108828/108828_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
109830 /data/backed_up/shared/xitchen_WM/results/109830/109830_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/109830/109830_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
110007 /data/backed_up/shared/xitchen_WM/results/110007/110007_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/110007/110007_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
110411 /data/backed_up/shared/xitchen_WM/results/110411/110411_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/110411/110411_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
110613 /data/backed_up/shared/xitchen_WM/results/110613/110613_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/110613/110613_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
111312 /data/backed_up/shared/xitchen_WM/results/111312/111312_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/111312/111312_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
111716 /data/backed_up/shared/xitchen_WM/results/111716/111716_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/111716/111716_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
112112 /data/backed_up/shared/xitchen_WM/results/112112/112112_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/112112/112112_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
112314 /data/backed_up/shared/xitchen_WM/results/112314/112314_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/112314/112314_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
112920 /data/backed_up/shared/xitchen_WM/results/112920/112920_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/112920/112920_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
113619 /data/backed_up/shared/xitchen_WM/results/113619/113619_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/113619/113619_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
113922 /data/backed_up/shared/xitchen_WM/results/113922/113922_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/113922/113922_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
114419 /data/backed_up/shared/xitchen_WM/results/114419/114419_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/114419/114419_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
115320 /data/backed_up/shared/xitchen_WM/results/115320/115320_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/115320/115320_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
116524 /data/backed_up/shared/xitchen_WM/results/116524/116524_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/116524/116524_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
117122 /data/backed_up/shared/xitchen_WM/results/117122/117122_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/117122/117122_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
118124 /data/backed_up/shared/xitchen_WM/results/118124/118124_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/118124/118124_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
118528 /data/backed_up/shared/xitchen_WM/results/118528/118528_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/118528/118528_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
118730 /data/backed_up/shared/xitchen_WM/results/118730/118730_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/118730/118730_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
118932 /data/backed_up/shared/xitchen_WM/results/118932/118932_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/118932/118932_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
119126 /data/backed_up/shared/xitchen_WM/results/119126/119126_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/119126/119126_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
120111 /data/backed_up/shared/xitchen_WM/results/120111/120111_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/120111/120111_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
121416 /data/backed_up/shared/xitchen_WM/results/121416/121416_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/121416/121416_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
122317 /data/backed_up/shared/xitchen_WM/results/122317/122317_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/122317/122317_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
122620 /data/backed_up/shared/xitchen_WM/results/122620/122620_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/122620/122620_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
122822 /data/backed_up/shared/xitchen_WM/results/122822/122822_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/122822/122822_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
123117 /data/backed_up/shared/xitchen_WM/results/123117/123117_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/123117/123117_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
123925 /data/backed_up/shared/xitchen_WM/results/123925/123925_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/123925/123925_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
124422 /data/backed_up/shared/xitchen_WM/results/124422/124422_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/124422/124422_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
124826 /data/backed_up/shared/xitchen_WM/results/124826/124826_FIRmodel_MNI_stats_GS_REML+tlrc${coef[$c]} /data/backed_up/shared/xitchen_WM/results/124826/124826_FIRmodel_MNI_stats_GS_REML+tlrc${tstat[$c]} \
-cio \
-missing_data 0 \
-model_outliers
done
