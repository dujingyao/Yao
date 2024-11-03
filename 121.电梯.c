#include<stdio.h>
int main()
{
	int N,i,sumtime=0,time,j;
	scanf("%d",&N);
	int a[N];
	for(i=0;i<N;i++){
		scanf("%d",&a[i]);
	}
	time=a[0]*6+5;
	sumtime+=time;
	for(j=0;j<N-1;j++){
		if(a[j]<a[j+1]) time=(a[j+1]-a[j])*6+5;
		if(a[j]>a[j+1]) time=(a[j]-a[j+1])*4+5;
		if(a[j]==a[j+1]) time=5;
		sumtime+=time;
	}
	printf("%d",sumtime);
	return 0;
}
