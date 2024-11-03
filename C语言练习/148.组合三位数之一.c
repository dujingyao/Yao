#include<stdio.h>
#include<math.h>
int f(int a,int b,int c);
int main()
{
	int i,x,b[13],j=0,t,k;
	for(i=10;i<sqrt(1000);i++){
		x=i*i;
		if(x%10!=x/10%10&&x/10%10!=x/100&&x/100!=x%10){
			b[j]=x;
			j++;
			t=j;
		}
	}
	for(i=0;i<t;i++){
		for(j=i;j<t;j++){
			for(k=j;k<t;k++){
				if(f(b[i],b[j],b[k])) printf("%d %d %d\n",b[i],b[j],b[k]);
			}
		}
  }
	
	
	return 0;
}
int f(int a,int b,int c){
	int i,d[10]={0};
	while(a){
		i=a%10;
		d[i]++;
		a/=10;
	}
	while(b){
		i=b%10;
		d[i]++;
		b/=10;
	}
	while(c){
		i=c%10;
		d[i]++;
		c/=10;
	}
	for(i=1;i<=9;i++){
		if(d[i]!=1) return 0;
	}
	return 1;
  }
