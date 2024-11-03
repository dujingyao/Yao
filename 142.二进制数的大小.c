#include<stdio.h>
#include<string.h>
int bToD(char str[]);
int main()
{
	char a[30],b[30],c[30];
	int x,y,z,t;
	scanf("%s %s %s",a,b,c);
	x=bToD(a),y=bToD(b),z=bToD(c);
	if(x>y){
		t=x;
		x=y;
		y=t;
	}
	if(x>z){
		t=x;
		x=z;
		z=t;
	}
	if(y>z){
		t=y;
		y=z;
		z=t;
	}
	printf("%d %d %d",x,y,z);
	
	
	return 0;
}
int bToD(char str[]){
	int x=0;
	for(int i=0;i<strlen(str);i++){
		x=x*2+(str[i]-'0');
	}
	return x;
}
