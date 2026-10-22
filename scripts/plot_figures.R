# Base R scientific graphics from the frozen participant-level analysis tables.
a <- commandArgs(trailingOnly=TRUE)
f <- a[1]
curves <- read.csv(file.path(f,'risk_coverage.csv'))
policies <- read.csv(file.path(f,'policy_points.csv'))
effects <- read.csv(file.path(f,'effects.csv'))
bins <- read.csv(file.path(f,'reliability.csv'))
methods <- c('deterministic','ensemble','dropout')
cols <- c('#333333','#1764A0','#AE5220')
labels <- c('Baseline entropy','Five-model ensemble','MC dropout (30 passes)')
parts <- c('internal_test','external_test')
titles <- c('Internal D2 (n=11)','External D1 (n=26)')
openfig <- function(name,leg=TRUE) {
 pdf(file.path(f,paste0(name,'.pdf')),width=10,height=4.8,pointsize=11,useDingbats=FALSE)
 if (leg) layout(matrix(c(1,2,3,3),nrow=2,byrow=TRUE),heights=c(.88,.12)) else par(mfrow=c(1,2))
 par(mar=c(4.3,4.8,2.6,.8),mgp=c(2.7,.8,0),tcl=-.25)
}
legendpanel <- function() {
 par(mar=c(0,0,0,0));plot.new();legend('center',legend=labels,col=cols,lty=1:3,pch=c(16,17,15),lwd=2,horiz=TRUE,bty='n',cex=.9)
}
openfig('Figure_1_risk_coverage')
for (j in 1:2) {
 plot(NA,xlim=c(0,1),ylim=c(0,1),xaxs='i',yaxs='i',xlab='Retained fraction',ylab='Mean participant 1-Dice',main=titles[j],bty='l')
 for (k in 1:3) {
  d <- curves[curves$cohort==parts[j]&curves$method==methods[k],]
  lines(d$coverage,d$risk,col=cols[k],lwd=1.7,lty=k)
  p <- policies[policies$cohort==parts[j]&policies$method==methods[k],]
  points(p$coverage,p$risk,col=cols[k],pch=c(16,17,15)[k],cex=.85)
 }
}
legendpanel();dev.off()
openfig('Figure_2_frozen_policy_effect',FALSE)
par(mar=c(4.8,6.4,2.6,.8))
bounds <- c(-.5,.5,effects$lower,effects$upper)
limits <- range(bounds,na.rm=TRUE)
for (j in 1:2) {
 plot(NA,xlim=limits,ylim=c(.5,3.5),xlab='Retained error minus overall error (95% CI)',ylab='',yaxt='n',main=titles[j],bty='l')
 axis(2,at=3:1,labels=c('Baseline','Ensemble','Dropout'),las=1,tick=FALSE)
 abline(v=0,col='grey65',lty=2)
 for (k in 1:3) {
  d <- effects[effects$cohort==parts[j]&effects$method==methods[k],]
  y <- 4-k
  if (is.na(d$estimate)) text(0,y,'No participants retained',cex=.8) else {
   if (!is.na(d$lower)&&!is.na(d$upper)) {
    segments(d$lower,y,d$upper,y,col=cols[k],lwd=1.8)
    segments(c(d$lower,d$upper),y-.055,c(d$lower,d$upper),y+.055,col=cols[k],lwd=1.3)
   }
   points(d$estimate,y,col=cols[k],pch=c(16,17,15)[k],cex=.85)
  }
 }
}
dev.off()
openfig('Figure_3_probability_reliability')
for (j in 1:2) {
 plot(NA,xlim=c(0,1),ylim=c(0,1),xaxs='i',yaxs='i',xlab='Mean predicted probability',ylab='Observed target fraction',main=titles[j],bty='l')
 abline(0,1,col='grey70',lty=2)
 for (k in 1:3) {
  d <- bins[bins$cohort==parts[j]&bins$method==methods[k],]
  lines(d$probability,d$observed,col=cols[k],lwd=1.7,lty=k,type='b',pch=c(16,17,15)[k],cex=.5)
 }
}
legendpanel();dev.off()
