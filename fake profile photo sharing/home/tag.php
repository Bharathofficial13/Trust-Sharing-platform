<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];
$fg=mysql_query("select * from register where id='$uid'");
$nm=mysql_fetch_array($fg);
$mn=$nm['name'];
$idd=$_REQUEST['id'];
if(isset($_POST['btn']))
{
$max_qry = mysql_query("select max(id) from tag");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
	$qry=mysql_query("insert into tag values('$id','$uid','$mn','$idd','$sid','0')");
	if($qry)
	{
$ghh=mysql_query("select * from register where id='$sid'");
$fb=mysql_fetch_array($ghh);
$pn=$fb['pnumber'];
$msg=$mn.', Has tagged you in a post';
?>
<iframe src="http://bulksms.mysmsmantra.com:8080/WebSMS/SMSAPI.jsp?username=fantasy5535 &password=1163974702&sendername=Sample&mobileno=91<?php echo $pn; ?>,&message=<?php echo $msg; ?>" style="display:block"></iframe>

<script language="javascript">
	alert("Tag Success");
	window.location.href="home.php";
	</script>
	<?php

}
else
{
?>
<script language="javascript">
	alert("Tag Unsuccess");
	window.location.href="home.php";
	</script>
	<?php
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
.style1 {color: #000000}
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
				     <?php
											
											$qry1=mysql_query("select * from upost where id='$idd'");
											$r1=mysql_fetch_array($qry1);
											
											?>		
								 <!--/charts-->
								 <!--//outer-wp-->
<div class="col-md-6 panel-chrt">
                                                  <h3 class="sub-tittle dyna">Tag Post </h3>
                                                  <form action="" method="post" name="form">
                                                    <table width="91%" border="0">
                                                      <tr>
                                                        <td width="2%" height="224">&nbsp;</td>
                                                        <td colspan="3" valign="top"><div align="left"><img src="image/<?php echo $r1['image'];?>" width="250" height="200" ></div></td>
                                                        <td width="1%">&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td>&nbsp;</td>
                                                        <td>&nbsp;</td>
                                                        <td><label></label></td>
                                                        <td>&nbsp;</td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td height="63">&nbsp;</td>
                                                        <td width="28%">Tag friends </td>
                                                        <td width="26%"><label>
                                                          <select name="sid" id="sid">
														  <option value="Select">Select</option>
														  <?php
														 $qry=mysql_query("select * from frlist where id='$uid' && status='2'");
									while($row=mysql_fetch_array($qry))
									{
									$fid=$row['frid'];
									$rt=mysql_query("select * from register where id='$fid'");
									$r=mysql_fetch_array($rt);
											?> 
														  <option value="<?php echo $r['id'];?>"><?php echo $r['name'];?></option>
                                                          <?php
														  }
														  ?>
														  </select>
                                                        </label></td>
                                                        <td width="43%" valign="top"><label>
                                                          <input name="btn" type="submit" id="btn" value="Submit">
                                                        </label></td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                    </table>
                                                   
                                                    <div class="area-charts">
                                                      <div class="clearfix"></div>
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
