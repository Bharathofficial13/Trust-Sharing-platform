<?php
include("dbconnect.php");
session_start();
extract($_POST);
echo $uid=$_SESSION['uid'];
if(isset($_POST['btn']))
{
if($gname=='Public'){






$max_qry = mysql_query("select max(id) from gpost");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
	$pid=$id;
$imgpath=$_FILES['file']['name'];
	$errors= array();
    $fname = $_FILES['file']['name'];
    $file_tmp =$_FILES['file']['tmp_name'];
    move_uploaded_file($_FILES['file']['tmp_name'],"image/".$fname);
	$date=date("d-m-y");
	$t=date("h:i:sa");
$qry=mysql_query("insert into gpost values('$id','$uid','$fname','$caption','$status','1','$date','$t')");


if($qry)
{

?>
<script language="javascript">
	alert("Success");
	
	</script>
	<?php

}

}else{


$max_qry = mysql_query("select max(id) from gpost");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
	$pid=$id;
$imgpath=$_FILES['file']['name'];
	$errors= array();
    $fname = $_FILES['file']['name'];
    $file_tmp =$_FILES['file']['tmp_name'];
    move_uploaded_file($_FILES['file']['tmp_name'],"image/".$fname);
	$date=date("d-m-y");
	$t=date("h:i:sa");
$qry=mysql_query("insert into gpost values('$id','$uid','$fname','$caption','$status','','$date','$t')");
$qry;
if($qry)
{

$qrt = mysql_query("select * from grouplist where gname='$gname'");
while ($fg = mysql_fetch_array($qrt)) {
    $frid = $fg['frid'];
    $uuid = $fg['uid'];

    if ($frid == $uid) {
        $frid = $uuid; // Change the value of $frid to $uuid when the condition is met
    }

    $max_qry = mysql_query("select max(id) from grpost");
    $max_row = mysql_fetch_array($max_qry);
    $id1 = $max_row['max(id)'] + 1;

    $qer = mysql_query("insert into grpost values('$id1','$pid','$uid','$frid','$gname','$fname','0')");
    $qer;
/*$nm=mysql_query("select * from grouplist where uid='$uid' && frid!='$uid'");
while($g=mysql_fetch_array($nm))
{
$fid=$g['frid'];
$max_qry = mysql_query("select max(id) from grpost");
	$max_row = mysql_fetch_array($max_qry); 
	$id11=$max_row['max(id)']+1;
if($fid!=$frid)
{	
$qer=mysql_query("insert into grpost values('$id11','$pid','$fid','$frid','$gname','$fname','0')");
$qer;
}
}*/

}



?>
<script language="javascript">
	alert("Success");
	//window.location.href="home.php";
	</script>
	<?php
//}

}
}

}

?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Facebook Home</title>
	<link href="css/bootstrap.min.css" rel="stylesheet">
	<link href="css/font-awesome.min.css" rel="stylesheet">
	<link href="css/datepicker3.css" rel="stylesheet">
	<link href="css/styles.css" rel="stylesheet">
	
	<!--Custom Font-->
	<link href="https://fonts.googleapis.com/css?family=Montserrat:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">
	<!--[if lt IE 9]>
	<script src="js/html5shiv.js"></script>
	<script src="js/respond.min.js"></script>
	<![endif]-->
    <style type="text/css">
<!--
.im {border-radius: 50%;
}
.im1 {  border-radius: 50%;
}
.im2 {  border-radius: 50%;
}
-->
    </style>
</head>
<body>
	<nav class="navbar navbar-custom navbar-fixed-top" role="navigation">
		<div class="container-fluid">
			<div class="navbar-header">
				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#sidebar-collapse"><span class="sr-only">Toggle navigation</span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span></button>
				<a class="navbar-brand" href="#"><span>Facebook</span>Home</a>
				<ul class="nav navbar-top-links navbar-right">
				</ul>
			</div>
		</div><!-- /.container-fluid -->
	</nav>
	<div id="sidebar-collapse" class="col-sm-3 col-lg-2 sidebar">
		<div class="profile-sidebar">
			<div class="profile-userpic">
			<?php
					$qry=mysql_query("select * from register where id='$uid'");
					$rw=mysql_fetch_array($qry); 
					?>
				
			</div>
			<div class="profile-usertitle">
				<div class="profile-usertitle-name"><?php echo $rw['name'];?></div>
				<div class="profile-usertitle-status"><span class="indicator label-success"></span>&nbsp;&nbsp;&nbsp;&nbsp;<?php echo $rw['name'];?></div>
			</div>
			<div class="clear"></div>
		</div>
		<div class="divider"></div>
		<form role="search">
			<div class="form-group">
				<input type="text" class="form-control" placeholder="Search">
			</div>
		</form>
		<ul class="nav menu">
			<li><a href="home.php"><i class="fa fa-home"></i> <span>Home</span></a></li>
										 <li><a href="frlist.php"><i class="fa fa-book"></i> <span>Friend List</span></a></li>
										<li><a href="list.php"><i class="fa fa-users"></i> <span>List</span></a></li>
									<li><a href="post.php"><i class=" fa fa-book"></i> <span>Post</span></a></li>
									<li><a href="accept.php"><i class="fa fa-users"></i> <span>Accept</span></a></li>
									<li><a href="group.php"><i class="fa fa-users"></i> <span>Create Group</span></a></li>
									<li><a href="grouppost.php"><i class="fa fa-users"></i> <span>Group Post</span></a></li>
									
									<li><a href="profile.php"><i class="fa fa-bell-o"></i> <span>Profile</span></a></li>
									<li><a href="index.php"><i class="fa fa-power-switch"></i> <span>Logout</span></a></li>
		</ul>
	</div><!--/.sidebar-->
		
	<div class="col-sm-9 col-sm-offset-3 col-lg-10 col-lg-offset-2 main">
		<div class="row">
			<ol class="breadcrumb">
				<li><a href="#">
					<em class="fa fa-home"></em>
				</a></li>
				<li class="active">Forms</li>
			</ol>
		</div><!--/.row-->
		
		<div class="row">
		  <div class="col-lg-12"></div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading"></div>
				  <div class="panel-body">
				     <form action="" method="post" enctype="multipart/form-data" name="form1">
                                  <div class="graph-visual tables-main">
                                    <h2 class="inner-tittle">New Post </h2>
                                    <div class="graph-form">
                                      <div class="form-body">
                                        <div class="form-group">
                                          <label for="exampleInputFile">File input</label>
                                          <label>
                                          <input type="file" name="file">
                                          </label>
                                        </div>
										<div class="form-group"> 
										<label for="exampleInputPassword1">Caption</label>
										<p>
										  <textarea name="caption" id="caption"></textarea>
										  </p>
										</div>
										<div class="form-group"> 
										<strong>
										<label for="exampleInputPassword1"></label>
										</strong>
										<p class="color-gray"><strong>Group Name </strong></p>
										<p class="color-gray"><strong>
										  <select name="gname" id="gname">
										    <option value="--Select--">--Select--</option>
											 <option value="Public">Public</option>
											<?php
											$qry=mysql_query("select DISTINCT gname from grouplist where uid='$uid' || frid='$uid'");
											while($row=mysql_fetch_array($qry))
											{
											?>
											<option value="<?php echo $row['gname'];?>"><?php echo $row['gname'];?></option>
											<?php
											}
											?>
								        </select>
										</strong></p>
										</div>
                                        <button type="submit" class="btn btn-default" name="btn">Submit</button>
                                      </div>
                                    </div>
                                    <h3 class="inner-tittle two">
                                      <label></label>
                                    </h3>
                                  </div>
                          </form>
                                          
				  </div>
				</div><!-- /.panel-->
			</div>
			<!-- /.panel-->
	  </div><!-- /.col-->
			<div class="col-sm-12">
				<p class="back-link">Lumino Theme by <a href="https://www.medialoot.com">Medialoot</a></p>
			</div>
</div><!-- /.row -->
	</div><!--/.main-->
	
<script src="js/jquery-1.11.1.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/chart.min.js"></script>
	<script src="js/chart-data.js"></script>
	<script src="js/easypiechart.js"></script>
	<script src="js/easypiechart-data.js"></script>
	<script src="js/bootstrap-datepicker.js"></script>
	<script src="js/custom.js"></script>
	
</body>
</html>
