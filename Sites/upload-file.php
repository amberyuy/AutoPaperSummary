<?php  
$file = $_FILES["file"];
move_uploaded_file($file["tmp_name"],"upload/".$file["name"]);

$output = shell_exec('/anaconda3/bin/python main1sent.py');
header("Refresh:8;url=Result.php");
    echo '<div style="margin-left:500px;margin-top:200px"><img src="load.gif" width="120" height="120"></div>';
    //echo "<img src=load.gif>";
?>

