# Makefile

# example usage:
# make proposal

proposal : docs/proposal_report.pdf
eda : eda/eda.pdf
report : docs/final_report.pdf
train_digit : outputs/digit.pt
train_odo : outputs/odo.pt
train :  train_digit train_odo
evaluate : outputs/Test_Data_Results.json outputs/Test_Data_Skipped.json
app : train_digit train_odo evaluate

# generate eda figures
tmp/proposal/image_quality_df.csv: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

tmp/proposal/metadata_df.csv: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

tmp/proposal/report_Fig2.png: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

tmp/proposal/report_Fig4.png: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

tmp/proposal/report_Fig5a.png: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

tmp/proposal/report_Fig5b.png: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

eda/eda.pdf: eda/eda.ipynb
	jupyter nbconvert --execute --to pdf eda/eda.ipynb

# write the proposal report
docs/proposal_report.pdf: tmp/proposal/image_quality_df.csv tmp/proposal/metadata_df.csv tmp/proposal/report_Fig2.png tmp/proposal/report_Fig4.png tmp/proposal/report_Fig5a.png tmp/proposal/report_Fig5b.png
	Rscript -e "rmarkdown::render('docs/proposal_report.Rmd')"

# generate model files
outputs/digit.pt : notebooks/train_digit_det.ipynb
	@read -p "Are you sure you want to delete the tmp/train_digit directory? [y/N]: " confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] && rm -rf tmp/train_digit || echo "Aborted"
	jupyter nbconvert --execute --to pdf --output ../outputs/`date +'%Y%m%d'`_digit_train_log.pdf notebooks/train_digit_det.ipynb

outputs/odo.pt : notebooks/train_odo_det.ipynb
	@read -p "Are you sure you want to delete the tmp/train_odo directory? [y/N]: " confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] && rm -rf tmp/train_odo || echo "Aborted"
	jupyter nbconvert --execute --to pdf --output ../outputs/`date +'%Y%m%d'`_odo_train_log.pdf notebooks/train_odo_det.ipynb

# evaluate Model
outputs/Test_Data_Results.json : outputs/digit.pt outputs/odo.pt
	python src/evaluate.py

outputs/Test_Data_Skipped.json : outputs/digit.pt outputs/odo.pt
	python src/evaluate.py

# generate final report figures
tmp/report/final_report_Fig3.png: notebooks/report_figures.ipynb outputs/Test_Data_Results.json
	jupyter nbconvert --execute --to pdf notebooks/report_figures.ipynb

tmp/report/final_report_Fig5.png: notebooks/report_figures.ipynb outputs/Test_Data_Results.json
	jupyter nbconvert --execute --to pdf notebooks/report_figures.ipynb

tmp/report/final_report_Fig6a.png: notebooks/report_figures.ipynb outputs/Test_Data_Results.json
	jupyter nbconvert --execute --to pdf notebooks/report_figures.ipynb

tmp/report/final_report_Fig6b.png: notebooks/report_figures.ipynb outputs/Test_Data_Results.json
	jupyter nbconvert --execute --to pdf notebooks/report_figures.ipynb

# write the final report
docs/final_report.pdf: tmp/report/final_report_Fig3.png tmp/report/final_report_Fig5.png tmp/report/final_report_Fig6a.png tmp/report/final_report_Fig6b.png 
	Rscript -e "rmarkdown::render('docs/final_report.Rmd')"

clean:
	rm -rf tmp/proposal
	rm -rf tmp/report
	rm -rf tmp/app
	rm tmp/yolov8?.pt
	@read -p "Are you sure you want to delete the tmp/train_digit directory? [y/N]: " confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] && rm -rf tmp/train_digit || echo "Aborted"
	@read -p "Are you sure you want to delete the tmp/train_odo directory? [y/N]: " confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] && rm -rf tmp/train_odo || echo "Aborted"
  