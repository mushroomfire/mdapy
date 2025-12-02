rm source -r
sphinx-apidoc -o source ../src/mdapy/

rm source/modules.rst
sed -i '/^Module contents/,$d' source/mdapy.rst
sed -i '1,5d' source/mdapy.rst
sed -i '1i Index\n======\n' source/mdapy.rst
echo "✅ Cleaned up source/mdapy.rst"

make clean && make html
