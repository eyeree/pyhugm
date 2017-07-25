#include "Lepton_I2C.h"
#include "stdio.h"

int main( int argc, char **argv )
{
	printf("performing ffc\n");	
	lepton_perform_ffc();
	return 0;
}

