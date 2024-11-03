#include<stdio.h>
int main()
{
	int a[110]={0};
	int num,max,i,find=0;
	while(scanf("%d",&num),num!=-1){
		a[num]++;
	}
	max=a[0];
	for(i=1;i<110;i++){
		if(max<a[i]) max=a[i];
	}
	for(i=0;i<110;i++){
		if(a[i]==max&&find==0){
			printf("%d",i);
			find=1;
		}else if(a[i]==max&&find!=0)
			printf(" %d",i);
	}
	
	return 0;
}
