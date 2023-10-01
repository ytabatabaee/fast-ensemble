require(ggplot2);require(reshape2);require(scales);require(ggpubr);require(tidyr)

#res_limit_exps= read.csv('res_limit_exps.csv')
#res_limit_exps= read.csv('res_limit_exps_leiden_cpm.csv')
#res_limit_exps= read.csv('res_limit_exps_leiden_cpm_varyres_tree.csv')
#res_limit_exps= read.csv('res_limit_exps_leiden_mod_tree.csv')
#res_limit_exps$partition <- factor(res_limit_exps$partition, levels = c("Leiden-CPM(r=0.0001)", "SC(np=10)+Leiden-CPM(r=0.0001)", "SC(np=50)+Leiden-CPM(r=0.0001)", "SC(np=100)+Leiden-CPM(r=0.0001)"))


give.n <- function(x){
  return(c(y = median(x)*1.05, label = length(x))) 
}


ggplot(aes(x= as.factor(res),y=cluster_size,fill=partition, color=partition), data=res_limit_exps)+
  #facet_wrap(~method,ncol=2)+
  geom_boxplot(outlier.size = 0)+
  stat_summary(fun.data = mean_sdl,position = position_dodge(width=0.75))+
  stat_summary(fun.data = give.n, geom = "text", vjust=-1.5, position = position_dodge(width=0.75), col="black", size=3)+
  scale_y_continuous(name="Cluster size distribution")+
  #scale_x_discrete(name="Number of cliques of size 10")+
  scale_x_discrete(name="Resolution value")+
  scale_fill_brewer(palette="Set2")+
  scale_color_brewer(palette = "Dark2")+
  theme_bw()
ggsave("res_limit_exps_lieden_cpm_varyres_tree.pdf",width=9,height =  4)



