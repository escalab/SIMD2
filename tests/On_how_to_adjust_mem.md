This method has not be tested remotely yet.

**Driver version: 470.86**  
**OS: Ubuntu 20.04.3 LTS**

You first need to "coolbits" to 28 [other may work, not tested]

``sudo nvidia-xconfig --cool-bits=28``
, then reboot your machine

## To list all supported memory clock mode / graphics mode:

``nvidia-smi --query-supported-clocks="mem" --format="csv"``

``nvidia-smi --query-supported-clocks="gr" --format="csv"``

[ there is not much freedom for memory clock, can only adjust in between provided mdes. ]

## To enable presistence mode (required for adjusting mem/graph clock):

``sudo nvidia-smi -pm 1``

[ can use 0 instead of 1 to disable, but suggest to reboot ]

## To adjust Graphcs clock:

``sudo nvidia-smi -lgc [CLOCK]``

## To adjust Mem clock:

``sudo nvidia-smi -lmc [CLOCK]``

[ you can enter any numbers but the tool will only change to closest mode listed in the previous steps ]


## To monitor your clocks:
``watch -n 1 nvidia-smi -q -d CLOCK``

## Kown Issue:
Overclocking doesn't seems to to be working on my tested cases.