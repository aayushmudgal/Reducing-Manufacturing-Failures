
library(data.table)

data <- fread("../input/train_numeric.csv",select = c("Id", "Response"));
data <- data[,c('Id', 'Response')];
data$nextId = data$Id + 1;
IdList <- data[data$nextId %in% data$Id];


data[Id %in% IdList$Id] -> current;
data[Id %in% IdList$nextId] -> nextId;

prob = table(current$Response,nextId$Response);

prob1 = prob[2,2]/(prob[2,1]+prob[2,2]);
prob2 = prob[2,1]/(prob[1,1]+prob[2,1]);

print("Percentage of defect after defect");
print(prob1);

print("Percentage of defect after no defect");
print(prob2);
