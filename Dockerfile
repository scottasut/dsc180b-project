ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable
FROM $BASE_CONTAINER
LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"
USER jovyan
RUN pip install --no-cache-dir zstandard kaggle praw sklearn pandas pytigergraph numpy seaborn matplotlib
CMD ["/bin/bash"]