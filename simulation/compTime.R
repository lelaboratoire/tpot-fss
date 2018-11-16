file.info("/Users/ttle/tpot-ds/simulation/pipelines_ds/")$ctime



filenames <- list.files("pipelines_ds", pattern="*.py", full.names=TRUE)
my.paths <- paste0(getwd(), "/", filenames)
stamps <- file.info(my.paths)$mtime
# stamps[2] - stamps[1]
diff(sort(stamps))
# str(file.info(my.path))
hist(as.numeric(diff(sort(stamps))))
boxplot(as.numeric(diff(sort(stamps))))
mean(diff(sort(stamps)))
sd(diff(sort(stamps)))
plot(diff(sort(stamps)))

filenames <- list.files("pipelines_reg", pattern="*.py", full.names=TRUE)
my.paths <- paste0(getwd(), "/", filenames)
stamps <- file.info(my.paths)$mtime
# stamps[2] - stamps[1]
diff(sort(stamps))
# str(file.info(my.path))
hist(as.numeric(diff(sort(stamps))))
boxplot(as.numeric(diff(sort(stamps))))
mean(diff(sort(stamps)))
sd(diff(sort(stamps)))
plot(diff(sort(stamps)))
