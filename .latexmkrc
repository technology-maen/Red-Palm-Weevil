$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';
$clean_ext = 'synctex.gz';

# Prevent latexmk from removing some intermediate files if you prefer to keep them
$clean_full = 0;
