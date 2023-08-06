
echo 'Press q to quit or space to take a picture'
python3 screenshotUSB.py -w
ls

echo 'Press q to quit, a to show the piture, r to remove it or s to save it in images'
while : ;
do
	read -n 1 k <&1
	echo ' '
if [[ $k == 'q' ]] ; then
	exit;
elif [[ $k == 'a' ]] ; then
	eog image.jpg
elif [[ $k == 'r' ]] ; then
	rm image.jpg
	ls
else 
	echo 'Wrong key !'
	echo 'Press q to quit, a to show the piture, r to remove it or s to save it in images'
fi
done


